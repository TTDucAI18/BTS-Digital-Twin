#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import glob as _glob
import shutil
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        # weights_only=False: required for checkpoints containing numpy scalars (optimizer states)
        # Compatible with PyTorch >= 2.6 where default changed from False to True
        try:
            (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint '{checkpoint}': {e}")
            print(f"[ERROR] Checkpoint may be corrupted (e.g. truncated by disk-full). Training from scratch.")
            first_iter = 0
            gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
            scene = Scene(dataset, gaussians)
            gaussians.training_setup(opt)
        else:
            gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 

    # ── TASK 2: Hybrid Depth Weight Scheduler (DA-v2 Two-Phase Strategy) ─────────
    # Phase 1 (0 – 5 000 iters)  : Full base_weight  → anchor Gaussians on BTS tower
    # Phase 2 (5 000 – 25 000)   : Linear decay → 0  → release for cable geometry
    # Phase 3 (25 000 – 30 000+) : weight = 0        → color-only, zero VRAM overhead
    def get_depth_weight(iteration, base_weight=0.1):
        if iteration <= 5_000:
            return base_weight
        elif iteration < 25_000:
            progress = (iteration - 5_000) / (25_000 - 5_000)
            # Không giảm về 0, giữ lại 5% trọng số để tránh floaters
            return base_weight * ((1.0 - progress) * 0.95 + 0.05)
        else:
            return base_weight * 0.05

    # Phục hồi Exposure Compensation (giúp giảm mây mờ / floaters do chênh lệch độ sáng giữa các ảnh từ camera drone)
    num_cams = max([c.uid for c in train_cameras]) + 1
    exposure_embedding = torch.nn.Embedding(num_cams, 3).cuda()
    exposure_embedding.weight.data.zero_() # Bắt đầu với bias = 0 (tương đương multiplier = 1.0)
    exposure_optimizer = torch.optim.Adam(exposure_embedding.parameters(), lr=0.01)

    # Extract validation set (chỉ để monitor — KHÔNG loại khỏi training)
    import random
    all_train_cams = scene.getTrainCameras()
    val_size = min(10, max(1, len(all_train_cams) // 10))
    # fixed seed for deterministic val set
    random.seed(42)
    val_indices = set(random.sample(range(len(all_train_cams)), val_size))
    val_cameras = [c for i, c in enumerate(all_train_cams) if i in val_indices]
    # Dùng 100% ảnh để train (không loại bỏ val set ra)
    train_cameras = all_train_cams

    viewpoint_stack = train_cameras.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", file=sys.stdout)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]  # TASK 1: use_trained_exp removed
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)  # giữ đồng bộ với viewpoint_stack

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, separate_sh=SPARSE_ADAM_AVAILABLE)  # TASK 1: use_trained_exp removed
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Áp dụng bù trừ phơi sáng cho từng ảnh
        # Điều này giúp mô hình không phải tạo ra sương mù/floaters để bù sáng
        cam_uid = torch.tensor([viewpoint_cam.uid], device="cuda", dtype=torch.long)
        exp_modifier = torch.exp(exposure_embedding(cam_uid)).squeeze(0).unsqueeze(-1).unsqueeze(-1)
        # Khóa lại ở [0, 1] trước khi tính loss
        image_adj = torch.clamp(image * exp_modifier, 0.0, 1.0)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image_adj, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image_adj.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image_adj, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # ── KHẮC PHỤC LỖI TOÁN HỌC: Depth Regularization (Chống Exploding Gradients) ────
        Ll1depth_pure = 0.0
        current_depth_weight = get_depth_weight(iteration, base_weight=opt.depth_weight_init)
        if opt.depth_weight_init > 0.0 and current_depth_weight > 0.0 and viewpoint_cam.invdepthmap is not None:
            rendered_linear_depth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            # CHỈNH SỬA QUAN TRỌNG:
            # 1. Biến mono_invdepth thành linear depth thay vì làm ngược lại.
            # 2. Đạo hàm của 1/x = -1/x^2 (gây bùng nổ gradient nếu tính nghịch đảo rendered_depth).
            # Do đó, bắt buộc phải đổi ground_truth về linear space.
            mono_linear_depth = 1.0 / mono_invdepth.clamp(min=1e-4)

            # Tính Smooth L1 (Huber Loss) để giảm sự nhạy cảm với các nhiễu lớn từ mono_depth
            diff = torch.nn.functional.smooth_l1_loss(
                rendered_linear_depth * depth_mask, 
                mono_linear_depth * depth_mask, 
                reduction='sum'
            )
            Ll1depth_pure = diff / (depth_mask.sum() + 1e-6)
            Ll1depth = current_depth_weight * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                epoch = iteration // len(train_cameras)
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Epoch": epoch})
                progress_bar.update(10)

            if iteration % 2000 == 0:
                free_space_gb = shutil.disk_usage(scene.model_path).free / (1024**3)
                if free_space_gb < 3.5:
                    print(f"\n[ITER {iteration}] WARNING: Disk space low ({free_space_gb:.2f} GB left). Stopping training early.")
                    torch.cuda.empty_cache()  # Prevent OOM before emergency save
                    scene.save(iteration)
                    new_ckpt_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                    torch.save((gaussians.capture(), iteration), new_ckpt_path)
                    
                    def _ckpt_iter(p):
                        try:
                            return int(os.path.basename(p).replace("chkpnt", "").replace(".pth", ""))
                        except:
                            return 0
                    all_ckpts = sorted(_glob.glob(os.path.join(scene.model_path, "chkpnt*.pth")), key=_ckpt_iter)
                    for old_ckpt in all_ckpts:
                        if old_ckpt != new_ckpt_path:
                            try:
                                os.remove(old_ckpt)
                            except Exception:
                                pass
                    break
                if wandb.run is not None:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/l1_loss": Ll1.item(),
                        "train/ssim": ssim_value.item(),
                        "train/epoch": epoch
                    }, step=iteration)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE), val_cameras)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # FIX BUG 4: Free VRAM before writing large PLY to disk.
                # At iter 30000, Gaussian count can be very large (>1M points).
                # Keeping optimizer tensors in VRAM while writing causes OOM (rc=137).
                torch.cuda.empty_cache()
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if gaussians.get_xyz.shape[0] < 15_000_000:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                    else:
                        print(f"\n[ITER {iteration}] WARNING: Max Gaussians reached ({gaussians.get_xyz.shape[0]}). Skipping densification to prevent OOM.")
                        # Still prune to remove transparent ones
                        prune_mask = (gaussians.get_opacity < 0.005).squeeze()
                        if size_threshold:
                            big_points_vs = gaussians.max_radii2D > size_threshold
                            big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * scene.cameras_extent
                            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                        gaussians.tmp_radii = radii
                        gaussians.prune_points(prune_mask)
                        gaussians.tmp_radii = None
                        torch.cuda.empty_cache()
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                exposure_optimizer.step()
                exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                new_ckpt_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), new_ckpt_path)
                # Disk cleanup: xóa tất cả checkpoint cũ hơn, chỉ giữ 1 checkpoint mới nhất
                # Giúp tiết kiệm disk trong giới hạn 20GB của Kaggle
                def _ckpt_iter(p):
                    try:
                        return int(os.path.basename(p).replace("chkpnt", "").replace(".pth", ""))
                    except:
                        return 0
                all_ckpts = sorted(_glob.glob(os.path.join(scene.model_path, "chkpnt*.pth")), key=_ckpt_iter)
                for old_ckpt in all_ckpts:
                    if old_ckpt != new_ckpt_path:
                        try:
                            os.remove(old_ckpt)
                            print("[ITER {}] Deleted old checkpoint: {}".format(iteration, os.path.basename(old_ckpt)))
                        except Exception as e:
                            print("[ITER {}] Could not delete {}: {}".format(iteration, os.path.basename(old_ckpt), e))

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND and not args.use_wandb:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
        
    if args.use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=os.path.basename(args.model_path), config=vars(args))
        
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, val_cameras):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'validation', 'cameras' : val_cameras})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                # Giới hạn số camera để tránh OOM khi Gaussian count cao
                eval_cams = config['cameras'][:5]
                l1_test = 0.0
                psnr_test = 0.0
                with torch.no_grad():
                    for idx, viewpoint in enumerate(eval_cams):
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        if wandb.run is not None and (idx < 5):
                            has_gt = (hasattr(viewpoint, 'original_image') 
                                      and viewpoint.original_image is not None
                                      and viewpoint.original_image.shape[0] == 3)
                            log_dict = {
                                f"{config['name']}_view_{viewpoint.image_name}/render": wandb.Image(image.permute(1, 2, 0).cpu().numpy()),
                            }
                            if has_gt:
                                log_dict[f"{config['name']}_view_{viewpoint.image_name}/ground_truth"] = wandb.Image(gt_image.permute(1, 2, 0).cpu().numpy())
                            wandb.log(log_dict, step=iteration)
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        del image, gt_image
                        torch.cuda.empty_cache()  # giải phóng fragment VRAM sau mỗi camera
                psnr_test /= len(eval_cams)
                l1_test /= len(eval_cams)          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb.run is not None:
                    wandb.log({
                        f"{config['name']}/l1_loss": l1_test,
                        f"{config['name']}/psnr": psnr_test
                    }, step=iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[25_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="bts-digital-twin")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (username or team)")
    args = parser.parse_args(sys.argv[1:])
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
