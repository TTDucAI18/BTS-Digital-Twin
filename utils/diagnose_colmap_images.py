import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scene"))

from colmap_loader import read_extrinsics_binary, read_extrinsics_text


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_colmap_images(sparse_dir):
    bin_path = sparse_dir / "images.bin"
    txt_path = sparse_dir / "images.txt"
    if bin_path.exists():
        return read_extrinsics_binary(str(bin_path))
    if txt_path.exists():
        return read_extrinsics_text(str(txt_path))
    raise FileNotFoundError(f"No images.bin/images.txt found in {sparse_dir}")


def casefold_map(paths):
    return {path.name.lower(): path for path in paths}


def main():
    parser = argparse.ArgumentParser(description="Diagnose COLMAP image-name mismatch.")
    parser.add_argument("--scene", required=True, help="Scene root or train root.")
    parser.add_argument("--images", default="images", help="Image folder name under train root.")
    parser.add_argument("--sparse", default="sparse/0", help="Sparse model folder under train root.")
    args = parser.parse_args()

    scene_root = Path(args.scene)
    train_root = scene_root / "train" if (scene_root / "train" / "sparse").exists() else scene_root
    sparse_dir = train_root / args.sparse
    images_dir = train_root / args.images

    colmap_images = load_colmap_images(sparse_dir)
    colmap_names = [image.name for image in colmap_images.values()]
    disk_images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    disk_by_name = casefold_map(disk_images)
    matched = [name for name in colmap_names if name.lower() in disk_by_name]
    missing = [name for name in colmap_names if name.lower() not in disk_by_name]
    colmap_by_name = {name.lower() for name in colmap_names}
    extra = [p.name for p in disk_images if p.name.lower() not in colmap_by_name]

    print(f"train_root: {train_root}")
    print(f"sparse_dir: {sparse_dir}")
    print(f"images_dir: {images_dir}")
    print(f"COLMAP camera poses: {len(colmap_names)}")
    print(f"image files: {len(disk_images)}")
    print(f"matched by filename: {len(matched)}")
    print(f"missing image files for COLMAP poses: {len(missing)}")
    print(f"extra image files without COLMAP pose: {len(extra)}")

    if missing:
        print("\nFirst missing:")
        for name in missing[:30]:
            print(f"  {name}")

    if extra:
        print("\nFirst extra:")
        for name in extra[:30]:
            print(f"  {name}")


if __name__ == "__main__":
    main()
