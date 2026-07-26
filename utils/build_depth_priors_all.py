"""Sequentially create quality-gated Depth Anything priors for all scenes."""

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_SCENES = ("bonsai", "chair", "HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--depth-anything-root", required=True, type=Path)
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--encoder", default="vitl", choices=("vits", "vitb", "vitl"))
    parser.add_argument("--input-size", default=518, type=int)
    args = parser.parse_args()
    utility_dir = Path(__file__).resolve().parent
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    for scene in scenes:
        train = args.data_root / scene / "train"
        images = train / "images"
        if not images.is_dir() or not (train / "sparse" / "0").is_dir():
            raise FileNotFoundError(f"Invalid scene layout: {train}")
        print(f"\n===== {scene}: Depth Anything =====", flush=True)
        subprocess.run([
            sys.executable, utility_dir / "generate_depth_anything.py",
            "--images", images, "--outdir", train / "depths_any",
            "--depth-anything-root", args.depth_anything_root,
            "--encoder", args.encoder, "--input-size", str(args.input_size),
        ], check=True)
        print(f"===== {scene}: COLMAP reliability gate =====", flush=True)
        subprocess.run([
            sys.executable, utility_dir / "build_reliable_depth_prior.py",
            "--base-dir", train, "--source-depths", train / "depths_any",
            "--out-depths", train / "depths_any_reliable",
        ], check=True)


if __name__ == "__main__":
    main()
