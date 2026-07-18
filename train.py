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
import tempfile
import zipfile
import time
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
from utils.training_utils import evenly_spaced_holdout_indices, foreground_edge_l1, foreground_weighted_l1, image_edge_l1, natural_image_key
try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False
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


def checkpoint_is_complete(path):
    """Validate a checkpoint archive before it replaces an older checkpoint."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < 1024:
            return False
        with zipfile.ZipFile(path, "r") as archive:
            return archive.testzip() is None
    except Exception:
        return False


def save_checkpoint_atomically(model_path, gaussians, iteration, include_optimizer):
    """Write, CRC-validate, then atomically publish a checkpoint."""
    final_path = os.path.join(model_path, f"chkpnt{iteration}.pth")
    fd, temporary_path = tempfile.mkstemp(prefix=f".chkpnt{iteration}.", suffix=".tmp", dir=model_path)
    os.close(fd)
    try:
        started = time.monotonic()
        torch.save((gaussians.capture(include_optimizer=include_optimizer), iteration), temporary_path)
        saved_after = time.monotonic()
        if not checkpoint_is_complete(temporary_path):
            raise RuntimeError(f"Checkpoint validation failed: {temporary_path}")
        validated_after = time.monotonic()
        os.replace(temporary_path, final_path)
        size_gb = os.path.getsize(final_path) / (1024 ** 3)
        print(
            f"[ITER {iteration}] Checkpoint published: {size_gb:.2f} GB "
            f"(save {saved_after - started:.1f}s, CRC {validated_after - saved_after:.1f}s, "
            f"optimizer_state={include_optimizer})"
        )
        return final_path
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def remove_superseded_checkpoints(model_path, keep_path):
    if not checkpoint_is_complete(keep_path):
        raise RuntimeError(f"Refusing to remove older checkpoints; invalid replacement: {keep_path}")
    for old_ckpt in _glob.glob(os.path.join(model_path, "chkpnt*.pth")):
        if os.path.abspath(old_ckpt) == os.path.abspath(keep_path):
            continue
        try:
            os.remove(old_ckpt)
            print("Deleted old checkpoint: {}".format(os.path.basename(old_ckpt)))
        except OSError as exc:
            print("Could not delete {}: {}".format(os.path.basename(old_ckpt), exc))

def training(dataset, opt, pipe, validation_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):

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

    # Keep a deterministic set of capture views for visual monitoring.  The
    # competition rewards reconstruction quality, not a held-out score, so by
    # default every available training image still contributes gradients.
    ordered_train_cams = sorted(scene.getTrainCameras(), key=lambda camera: natural_image_key(camera.image_name))
    val_indices = evenly_spaced_holdout_indices(len(ordered_train_cams), args.validation_fraction)
    val_cameras = [camera for i, camera in enumerate(ordered_train_cams) if i in val_indices]
    if args.validation_holdout:
        train_cameras = [camera for i, camera in enumerate(ordered_train_cams) if i not in val_indices]
        validation_name = "validation"
        train_mode = "holdout"
    else:
        train_cameras = ordered_train_cams
        validation_name = "fixed_train_monitor"
        train_mode = "full-data"
    print(
        f"Fixed monitor: {len(val_cameras)}/{len(ordered_train_cams)} views "
        f"({100 * len(val_cameras) / len(ordered_train_cams):.1f}%), mode={train_mode}."
    )

    # Phục hồi Exposure Compensation (giúp giảm mây mờ / floaters do chênh lệch độ sáng giữa các ảnh từ camera drone)
    num_cams = max([c.uid for c in train_cameras]) + 1
    exposure_embedding = torch.nn.Embedding(num_cams, 3).cuda()
    exposure_embedding.weight.data.zero_() # Bắt đầu với bias = 0 (tương đương multiplier = 1.0)
    exposure_optimizer = torch.optim.Adam(exposure_embedding.parameters(), lr=0.01)
    if not opt.use_exposure_compensation:
        # Exposure is neither saved in a checkpoint nor defined for unseen test
        # poses.  Do not let it inflate train-only quality by default.
        exposure_optimizer = None

    viewpoint_stack = train_cameras.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    epoch_loss_sum = 0.0
    epoch_l1_sum = 0.0
    epoch_ssim_sum = 0.0
    epoch_samples = 0
    completed_epochs = 0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc=args.progress_name, file=sys.stdout, dynamic_ncols=True)
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

        # With exposure compensation disabled, retain the rasterizer output
        # directly.  Clamping it would zero gradients at saturated edges, where
        # sharp BTS silhouettes and cables need the strongest correction.
        if exposure_optimizer is None:
            image_for_loss = image
        else:
            cam_uid = torch.tensor([viewpoint_cam.uid], device="cuda", dtype=torch.long)
            exp_modifier = torch.exp(exposure_embedding(cam_uid)).squeeze(0).unsqueeze(-1).unsqueeze(-1)
            image_for_loss = torch.clamp(image * exp_modifier, 0.0, 1.0)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        fg_mask = viewpoint_cam.foreground_mask.cuda() if viewpoint_cam.foreground_mask is not None else None
        if opt.foreground_loss_weight > 0.0 and fg_mask is not None:
            Ll1 = foreground_weighted_l1(image_for_loss, gt_image, fg_mask, opt.foreground_loss_weight)
        else:
            Ll1 = l1_loss(image_for_loss, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image_for_loss.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image_for_loss, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        foreground_edge_loss = 0.0
        if opt.foreground_edge_loss_weight > 0.0 and fg_mask is not None:
            foreground_edge_loss = foreground_edge_l1(image_for_loss, gt_image, fg_mask)
            loss += opt.foreground_edge_loss_weight * foreground_edge_loss

        image_edge_loss = 0.0
        if opt.image_edge_loss_weight > 0.0:
            image_edge_loss = image_edge_l1(image_for_loss, gt_image)
            loss += opt.image_edge_loss_weight * image_edge_loss

        # ── KHẮC PHỤC LỖI TOÁN HỌC: Depth Regularization (Chống Exploding Gradients) ────
        Ll1depth_pure = 0.0
        current_depth_weight = get_depth_weight(iteration, base_weight=opt.depth_weight_init)
        if opt.depth_weight_init > 0.0 and current_depth_weight > 0.0 and viewpoint_cam.invdepthmap is not None and viewpoint_cam.depth_reliable:
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
            epoch_loss_sum += loss.item()
            epoch_l1_sum += Ll1.item()
            epoch_ssim_sum += ssim_value.item()
            epoch_samples += 1
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                epoch = iteration // len(train_cameras)
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Epoch": epoch})
                progress_bar.update(10)

            if WANDB_FOUND and wandb.run is not None and iteration % args.wandb_log_interval == 0:
                wandb.log({
                    "iteration": iteration,
                    # These are deliberately named as single-view values.
                    # They vary with the sampled camera and are not a
                    # convergence criterion on their own.
                    "train/view_loss": loss.item(),
                    "train/view_l1_loss": Ll1.item(),
                    "train/view_ssim": ssim_value.item(),
                    "train/loss": loss.item(),
                    "train/l1_loss": Ll1.item(),
                    "train/ssim": ssim_value.item(),
                    "train/depth_loss": Ll1depth,
                    "train/depth_loss_raw": Ll1depth_pure.item() if hasattr(Ll1depth_pure, "item") else float(Ll1depth_pure),
                    "train/depth_weight": current_depth_weight,
                    "train/foreground_edge_loss": (
                        foreground_edge_loss.item()
                        if hasattr(foreground_edge_loss, "item") else float(foreground_edge_loss)
                    ),
                    "train/foreground_edge_weight": opt.foreground_edge_loss_weight,
                    "train/image_edge_loss": (
                        image_edge_loss.item()
                        if hasattr(image_edge_loss, "item") else float(image_edge_loss)
                    ),
                    "train/image_edge_weight": opt.image_edge_loss_weight,
                    "train/epoch": epoch,
                    "scene/total_points": gaussians.get_xyz.shape[0],
                }, step=iteration)

            # A camera stack is consumed without replacement.  Log its mean
            # once per full pass so WandB exposes a stable convergence signal.
            if not viewpoint_stack and epoch_samples:
                completed_epochs += 1
                if WANDB_FOUND and wandb.run is not None:
                    wandb.log({
                        "iteration": iteration,
                        "train_epoch/loss": epoch_loss_sum / epoch_samples,
                        "train_epoch/l1_loss": epoch_l1_sum / epoch_samples,
                        "train_epoch/ssim": epoch_ssim_sum / epoch_samples,
                        "train_epoch/index": completed_epochs,
                    }, step=iteration)
                epoch_loss_sum = 0.0
                epoch_l1_sum = 0.0
                epoch_ssim_sum = 0.0
                epoch_samples = 0

            stop_after_checkpoint = False
            if iteration % args.disk_check_interval == 0:
                free_space_gb = shutil.disk_usage(scene.model_path).free / (1024**3)
                if free_space_gb < args.min_free_disk_gb:
                    print(f"\n[ITER {iteration}] Disk free {free_space_gb:.2f} GB < {args.min_free_disk_gb:.2f} GB. Saving a verified checkpoint before stopping.")
                    stop_after_checkpoint = True
                if args.stop_at_unix_time > 0 and time.time() >= args.stop_at_unix_time:
                    print(f"\n[ITER {iteration}] Runtime deadline reached. Saving a verified checkpoint before stopping.")
                    stop_after_checkpoint = True
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), validation_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE), val_cameras, validation_name)
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
                    size_threshold = (
                        opt.max_screen_size
                        if iteration > opt.opacity_reset_interval and opt.max_screen_size > 0
                        else None
                    )
                    if opt.max_gaussians <= 0 or gaussians.get_xyz.shape[0] < opt.max_gaussians:
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                            size_threshold, radii, max_points=opt.max_gaussians
                        )
                    else:
                        print(f"\n[ITER {iteration}] WARNING: Max Gaussians reached ({gaussians.get_xyz.shape[0]}/{opt.max_gaussians}). Skipping densification to prevent OOM.")
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
                
                reset_is_due = (
                    opt.opacity_reset_interval > 0
                    and iteration % opt.opacity_reset_interval == 0
                )
                reset_is_allowed = (
                    opt.opacity_reset_until_iter <= 0
                    or iteration <= opt.opacity_reset_until_iter
                )
                if (reset_is_due and reset_is_allowed) or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                if exposure_optimizer is not None:
                    exposure_optimizer.step()
                    exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration in checkpoint_iterations or stop_after_checkpoint:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                try:
                    if args.checkpoint_stagger_seconds:
                        print(
                            f"[ITER {iteration}] Staggering checkpoint I/O for "
                            f"{args.checkpoint_stagger_seconds}s."
                        )
                        time.sleep(args.checkpoint_stagger_seconds)
                    new_ckpt_path = save_checkpoint_atomically(
                        scene.model_path,
                        gaussians,
                        iteration,
                        include_optimizer=args.checkpoint_optimizer_state,
                    )
                except Exception as exc:
                    print("[ITER {}] Checkpoint save failed: {}. Previous checkpoint is preserved.".format(iteration, exc))
                    raise
                # Disk cleanup: xóa tất cả checkpoint cũ hơn, chỉ giữ 1 checkpoint mới nhất
                # Giúp tiết kiệm disk trong giới hạn 20GB của Kaggle
                def _ckpt_iter(p):
                    try:
                        return int(os.path.basename(p).replace("chkpnt", "").replace(".pth", ""))
                    except:
                        return 0
                all_ckpts = sorted(_glob.glob(os.path.join(scene.model_path, "chkpnt*.pth")), key=_ckpt_iter)
                # Keep explicit milestone checkpoints (30k/35k/40k in the
                # notebook) for recovery/model selection.  A checkpoint
                # created by a disk/deadline stop remains protected through
                # ``new_ckpt_path`` even when it is not a scheduled milestone.
                protected_checkpoint_iterations = set(checkpoint_iterations)
                for old_ckpt in all_ckpts:
                    old_iteration = _ckpt_iter(old_ckpt)
                    if old_ckpt != new_ckpt_path and old_iteration not in protected_checkpoint_iterations:
                        try:
                            os.remove(old_ckpt)
                            print("[ITER {}] Deleted old checkpoint: {}".format(iteration, os.path.basename(old_ckpt)))
                        except Exception as e:
                            print("[ITER {}] Could not delete {}: {}".format(iteration, os.path.basename(old_ckpt), e))

                if stop_after_checkpoint:
                    print(f"[ITER {iteration}] Stopped cleanly after publishing checkpoint.")
                    break

def prepare_output_and_logger(args):    
    if args.use_wandb and not WANDB_FOUND:
        sys.exit("WandB logging requested but package 'wandb' is not installed.")

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
        # Use explicit wandb_name if provided (e.g. from kaggle_notebook.py),
        # otherwise fall back to the model directory basename.
        run_name = getattr(args, "wandb_name", None) or os.path.basename(args.model_path)
        # Retry loop: two parallel subprocesses racing to init the wandb-service
        # can cause the first one to fail silently.  Retry up to 3 times with a
        # brief back-off to survive transient socket / service-start conflicts.
        wandb_initialized = False
        for _wandb_attempt in range(3):
            try:
                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=run_name,
                    config=vars(args),
                    settings=wandb.Settings(start_method="thread"),
                )
                wandb_initialized = wandb.run is not None
                if wandb_initialized:
                    break
                raise RuntimeError("wandb.init returned without an active run")
            except Exception as _wandb_exc:
                if _wandb_attempt == 2:
                    print(f"[WandB] Init failed after 3 attempts: {_wandb_exc}. Continuing without WandB.")
                    break
                _delay = 5 * (2 ** _wandb_attempt)
                print(f"[WandB] Init attempt {_wandb_attempt + 1} failed ({_wandb_exc}); retrying in {_delay}s...")
                time.sleep(_delay)
        if wandb_initialized:
            wandb.define_metric("train/*", step_metric="iteration")
            wandb.define_metric("train_epoch/*", step_metric="iteration")
            wandb.define_metric("scene/*", step_metric="iteration")
            wandb.log({"run/started": 1, "iteration": 0}, step=0)

            # Upload configuration file
            wandb.save(os.path.join(args.model_path, "cfg_args"), base_path=args.model_path)

            # Upload the continuous train log file if provided by the environment
            wandb_log_file = os.getenv("WANDB_LOG_FILE")
            if wandb_log_file:
                wandb.save(wandb_log_file, base_path=os.path.dirname(wandb_log_file), policy="live")
        
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, validation_iterations, scene : Scene, renderFunc, renderArgs, val_cameras, validation_name):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Kaggle test poses intentionally have no reference images.  Only the held
    # out train views can produce meaningful image-space evaluation metrics.
    if iteration in validation_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': validation_name, 'cameras': val_cameras},)

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                eval_cams = config['cameras']
                l1_test = 0.0
                psnr_test = 0.0
                with torch.no_grad():
                    for idx, viewpoint in enumerate(eval_cams):
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if validation_iterations and iteration == validation_iterations[0]:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        if WANDB_FOUND and wandb.run is not None and (idx < 5):
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
                if WANDB_FOUND and wandb.run is not None:
                    wandb.log({
                        "iteration": iteration,
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
    parser.add_argument(
        "--validation_iterations",
        nargs="+",
        type=int,
        default=[],
        help="Iterations for held-out training-view evaluation; hidden test poses are never evaluated.",
    )
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.05,
        help="Fraction of train images selected evenly as a fixed monitor set.",
    )
    parser.add_argument(
        "--validation_holdout",
        action="store_true",
        help="Exclude fixed validation views from training. Disabled by default to maximise reconstruction coverage.",
    )
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument(
        "--progress_name",
        type=str,
        default="Training progress",
        help="Scene/GPU label displayed by tqdm when subprocess output is streamed.",
    )
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[25_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument(
        "--checkpoint_optimizer_state",
        action="store_true",
        help="Include Adam state for an exact resume; substantially increases checkpoint I/O.",
    )
    parser.add_argument(
        "--checkpoint_stagger_seconds",
        type=float,
        default=0.0,
        help="Delay each checkpoint to avoid simultaneous multi-GPU disk I/O.",
    )
    parser.add_argument(
        "--stop_at_unix_time",
        type=float,
        default=0.0,
        help="Unix timestamp at which to checkpoint and stop cleanly (0 disables the deadline).",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="bts-digital-twin")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (username or team)")
    parser.add_argument("--wandb_name", type=str, default=None, help="Explicit WandB run name; overrides the model_path basename.")
    parser.add_argument("--wandb_log_interval", type=int, default=100, help="Log scalar metrics to wandb every N iterations")
    parser.add_argument("--min_free_disk_gb", type=float, default=2.0, help="Stop training when free disk drops below this value")
    parser.add_argument("--disk_check_interval", type=int, default=100, help="Check free disk every N iterations")
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
    training(lp.extract(args), op.extract(args), pp.extract(args), args.validation_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    if WANDB_FOUND and wandb.run is not None:
        wandb.finish()

    # All done
    print("\nTraining complete.")
