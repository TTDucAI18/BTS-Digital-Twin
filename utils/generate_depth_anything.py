"""Generate 16-bit relative inverse-depth maps for the 3DGS depth loader.

Depth Anything V2's sample ``run.py`` emits an 8-bit visualisation.  This
utility writes a single-channel uint16 map instead, matching both
``camera_utils.py`` and ``make_depth_scale.py`` (which divide by 2**16).
Each map is independently normalised because the subsequent COLMAP fit stores
the per-view scale and offset in ``depth_params.json``.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Write Depth Anything V2 uint16 inverse-depth maps.")
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--depth-anything-root", required=True, type=Path)
    parser.add_argument("--encoder", choices=MODEL_CONFIGS, default="vitl")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input_size <= 0:
        raise ValueError("--input-size must be positive")
    sys.path.insert(0, str(args.depth_anything_root))
    from depth_anything_v2.dpt import DepthAnythingV2

    checkpoint = args.depth_anything_root / "checkpoints" / f"depth_anything_v2_{args.encoder}.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    sources = sorted(
        path for path in args.images.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not sources:
        raise RuntimeError(f"No supported images in {args.images}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device).eval()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(sources, start=1):
        destination = args.outdir / f"{source.stem}.png"
        if destination.exists() and not args.overwrite:
            print(f"[{index}/{len(sources)}] keeping {destination.name}")
            continue
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unreadable image: {source}")
        inverse_depth = model.infer_image(image, args.input_size)
        lo, hi = np.percentile(inverse_depth, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise RuntimeError(f"Degenerate depth prediction for {source}")
        # Reserve zero for invalid pixels; all generated pixels remain valid.
        encoded = np.clip((inverse_depth - lo) / (hi - lo), 0.0, 1.0)
        encoded = np.rint(encoded * 65534.0 + 1.0).astype(np.uint16)
        if not cv2.imwrite(str(destination), encoded):
            raise RuntimeError(f"Could not write {destination}")
        print(f"[{index}/{len(sources)}] wrote {destination.name}")


if __name__ == "__main__":
    main()
