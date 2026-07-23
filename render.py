import os
from argparse import ArgumentParser
from os import makedirs

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False


def view_opacity_mask(view, gaussians, near_camera_culls):
    """Return a per-view keep mask for explicitly configured near floaters.

    Rules have the form ``image_name:near_distance:scale_to_distance``.  A
    Gaussian is suppressed only for that exact output image when it lies
    within ``near_distance`` of the virtual camera, or when its largest 3D
    scale consumes more than ``scale_to_distance`` of its camera distance.
    The latter catches a giant splat whose centre is just outside the near
    radius.  A non-positive scale ratio disables that secondary rule.
    """
    rule = near_camera_culls.get(getattr(view, "image_name", ""))
    if rule is None:
        return None
    near_distance, scale_to_distance = rule
    xyz = gaussians.get_xyz
    distance = torch.linalg.vector_norm(xyz - view.camera_center.unsqueeze(0), dim=1)
    remove = distance < near_distance
    if scale_to_distance > 0.0:
        max_scale = gaussians.get_scaling.max(dim=1).values
        remove |= max_scale > distance.clamp_min(1e-4) * scale_to_distance
    removed = int(remove.sum().item())
    print(
        f"[floater-cull] {view.image_name}: removed {removed}/{len(remove)} "
        f"Gaussians (near<{near_distance:g}, scale/distance>{scale_to_distance:g})."
    )
    return ~remove


def render_view(view, gaussians, pipeline, background, separate_sh, scales, near_camera_culls):
    """Supersample a view, falling back to native resolution on render OOM."""
    original_width, original_height = view.image_width, view.image_height
    accumulated = None
    active_scales = scales
    opacity_mask = view_opacity_mask(view, gaussians, near_camera_culls)
    try:
        for scale in active_scales:
            view.image_width = int(original_width * scale)
            view.image_height = int(original_height * scale)
            image = render(
                view, gaussians, pipeline, background, separate_sh=separate_sh,
                opacity_mask=opacity_mask,
            )["render"]
            if scale != 1.0:
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=(original_height, original_width),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)
            if accumulated is None:
                accumulated = image
            else:
                accumulated += image
                del image
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or active_scales == [1.0]:
            raise
        print(f"[WARN] Render OOM for {getattr(view, 'image_name', '?')} at scales {active_scales}; retrying at 1.0.")
        if accumulated is not None:
            del accumulated
        torch.cuda.empty_cache()
        view.image_width, view.image_height = original_width, original_height
        active_scales = [1.0]
        accumulated = render(
            view, gaussians, pipeline, background, separate_sh=separate_sh,
            opacity_mask=opacity_mask,
        )["render"]
    finally:
        view.image_width, view.image_height = original_width, original_height

    return (accumulated / len(active_scales)).clamp(0.0, 1.0)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, separate_sh, ensemble_scales, save_gt, near_camera_culls):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    makedirs(render_path, exist_ok=True)
    if save_gt:
        makedirs(gts_path, exist_ok=True)

    # Reruns must not inherit files from an older test-poses list.  The
    # notebook validates exact names, so a stale render here would otherwise
    # make a repaired scene fail submission validation forever.
    for directory in (render_path, gts_path if save_gt else None):
        if directory is None:
            continue
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.lower().endswith((".png", ".jpg", ".jpeg")):
                os.unlink(entry.path)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        final_rendering = render_view(
            view, gaussians, pipeline, background, separate_sh, ensemble_scales,
            near_camera_culls,
        )
        out_name = getattr(view, "image_name", None) or f"{idx:05d}.jpg"
        render_file_path = os.path.join(render_path, out_name)
        rendering_np = final_rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        # Preserve thin coloured antenna and cable detail before the submission
        # packaging stage applies its size limit.
        Image.fromarray(rendering_np).save(render_file_path, format="JPEG", quality=95, subsampling=0, optimize=True)

        if save_gt:
            gt = view.original_image[0:3, :, :]
            gt_np = gt.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(gt_np).save(os.path.join(gts_path, out_name), format="JPEG", quality=95, subsampling=0, optimize=True)
            del gt
        del final_rendering


def render_sets(dataset, iteration, pipeline, skip_train, skip_test, separate_sh, ensemble_scales, near_camera_culls):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, separate_sh, ensemble_scales, save_gt=True, near_camera_culls=near_camera_culls)
        if not skip_test:
            # Test poses have no images; do not waste disk on black placeholders.
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, separate_sh, ensemble_scales, save_gt=False, near_camera_culls=near_camera_culls)


def parse_near_camera_culls(values, parser):
    rules = {}
    for value in values:
        try:
            image_name, near_distance, scale_to_distance = value.rsplit(":", 2)
            near_distance = float(near_distance)
            scale_to_distance = float(scale_to_distance)
        except ValueError:
            parser.error(
                "--near_camera_cull must be image_name:near_distance:scale_to_distance"
            )
        if not image_name or near_distance <= 0.0 or scale_to_distance < 0.0:
            parser.error("--near_camera_cull requires a name, near_distance > 0, and scale_to_distance >= 0")
        rules[image_name] = (near_distance, scale_to_distance)
    return rules


if __name__ == "__main__":
    parser = ArgumentParser(description="Render trained BTS Gaussian splats")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ensemble_scales", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--near_camera_cull", action="append", default=[], metavar="RULE",
        help="Per-image inference cull: image_name:near_distance:scale_to_distance",
    )
    args = get_combined_args(parser)
    # Training persists this flag in cfg_args to save memory.  Rendering must
    # load hidden poses to create the competition submission.
    args.skip_test_poses = False
    if not args.ensemble_scales or any(scale < 1.0 for scale in args.ensemble_scales):
        parser.error("--ensemble_scales must contain one or more values >= 1.0")
    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    near_camera_culls = parse_near_camera_culls(args.near_camera_cull, parser)
    if near_camera_culls:
        print(f"Configured targeted floater culls: {near_camera_culls}")
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.ensemble_scales, near_camera_culls)
