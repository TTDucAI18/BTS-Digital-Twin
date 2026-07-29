"""Build a camera-consistent pinhole copy of a BTS/Kaggle scene.

The supplied BTS sparse models use COLMAP ``SIMPLE_RADIAL`` cameras while the
competition test poses contain only pinhole intrinsics.  The rasterizer cannot
apply radial distortion, so training against the original distorted frames
silently mixes two camera models.  This tool keeps the test-pose coordinate
system (same resolution and K), undistorts the training images and masks, and
rewrites ``cameras.bin`` as ``PINHOLE``.

It intentionally never changes the source scene.  It also does not copy
monocular depths: their pixel coordinates would need the same transformation,
and the high-quality BTS profile disables those noisy priors anyway.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from utils.read_write_model import Camera, read_cameras_binary, write_cameras_binary


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MASK_DIRECTORIES = ("foreground_masks", "masks", "mask", "foreground")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original scene root containing train/ and test/.")
    parser.add_argument("--destination", type=Path, required=True, help="New writable scene root.")
    parser.add_argument(
        "--jpeg-quality", type=int, default=100,
        help="JPEG quality for remapped JPG frames (1..100; default: 100).",
    )
    return parser.parse_args()


def training_root(scene: Path) -> Path:
    nested = scene / "train"
    return nested if (nested / "sparse" / "0").is_dir() else scene


def output_training_root(source_scene: Path, destination_scene: Path) -> Path:
    return destination_scene / "train" if (source_scene / "train").is_dir() else destination_scene


def camera_matrix(camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    if camera.model == "SIMPLE_RADIAL":
        focal, cx, cy, k1 = map(float, camera.params)
        K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        distortion = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return K, distortion
    if camera.model == "RADIAL":
        focal, cx, cy, k1, k2 = map(float, camera.params)
        K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        distortion = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
        return K, distortion
    raise ValueError(f"Expected SIMPLE_RADIAL or RADIAL camera, got {camera.model} for id={camera.id}.")


def pinhole_camera(camera: Camera) -> Camera:
    K, _ = camera_matrix(camera)
    return Camera(
        id=camera.id,
        model="PINHOLE",
        width=camera.width,
        height=camera.height,
        params=np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64),
    )


def build_maps(camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    K, distortion = camera_matrix(camera)
    # Keep K and the full canvas exactly unchanged.  That is the pinhole K
    # written in test_poses.csv, unlike getOptimalNewCameraMatrix which would
    # crop/shift the image and require unavailable test-image remapping.
    return cv2.initUndistortRectifyMap(
        K, distortion, np.eye(3), K, (camera.width, camera.height), cv2.CV_32FC1
    )


def image_camera_map(cameras: dict[int, Camera]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {camera_id: build_maps(camera) for camera_id, camera in cameras.items()}


def read_image(path: Path, unchanged: bool) -> np.ndarray:
    flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise RuntimeError(f"Could not decode {path}")
    return image


def write_image(path: Path, image: np.ndarray, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params: list[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    if not cv2.imwrite(str(path), image, params):
        raise RuntimeError(f"Could not write {path}")


def copy_sparse_without_camera(source_sparse: Path, destination_sparse: Path) -> None:
    destination_sparse.mkdir(parents=True, exist_ok=True)
    for path in source_sparse.iterdir():
        if path.name in {"cameras.bin", "cameras.txt"}:
            continue
        if path.is_file():
            shutil.copy2(path, destination_sparse / path.name)


def copy_test_metadata(source_scene: Path, destination_scene: Path) -> None:
    source_test = source_scene / "test"
    if not source_test.is_dir():
        return
    destination_test = destination_scene / "test"
    destination_test.mkdir(parents=True, exist_ok=True)
    for path in source_test.iterdir():
        if path.is_file():
            shutil.copy2(path, destination_test / path.name)


def image_to_camera_ids(source_sparse: Path) -> dict[str, int]:
    # ``images.bin`` is intentionally not rewritten: 3DGS reads only the
    # camera id, pose and name.  Point tracks are irrelevant during training.
    # Import the lightweight loader from the repository rather than invoking
    # COLMAP, which keeps this preprocessing portable to Kaggle.
    from scene.colmap_loader import read_extrinsics_binary

    return {
        image.name: image.camera_id
        for image in read_extrinsics_binary(str(source_sparse / "images.bin")).values()
    }


def remap_directory(
    source_dir: Path,
    destination_dir: Path,
    name_to_camera: dict[str, int],
    maps: dict[int, tuple[np.ndarray, np.ndarray]],
    jpeg_quality: int,
    is_mask: bool,
) -> int:
    if not source_dir.is_dir():
        return 0
    count = 0
    for source_path in sorted(source_dir.iterdir()):
        if not source_path.is_file() or source_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        camera_id = name_to_camera.get(source_path.name)
        # Masks can legitimately have another extension.  They match images by
        # stem, so resolve their source camera name in that case.
        if camera_id is None and is_mask:
            candidates = [name for name in name_to_camera if Path(name).stem == source_path.stem]
            if len(candidates) == 1:
                camera_id = name_to_camera[candidates[0]]
        if camera_id is None:
            print(f"[WARN] Skipping {source_path.name}: no matching COLMAP image.")
            continue
        image = read_image(source_path, unchanged=is_mask)
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        remapped = cv2.remap(
            image, maps[camera_id][0], maps[camera_id][1], interpolation=interpolation,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        write_image(destination_dir / source_path.name, remapped, jpeg_quality)
        count += 1
    return count


def prepare(source_scene: Path, destination_scene: Path, jpeg_quality: int) -> None:
    source_scene = source_scene.resolve()
    source_train = training_root(source_scene)
    source_sparse = source_train / "sparse" / "0"
    if not (source_train / "images").is_dir() or not (source_sparse / "cameras.bin").is_file():
        raise FileNotFoundError(f"{source_scene} is not a supported binary-COLMAP BTS scene.")
    if destination_scene.exists():
        marker = destination_scene / ".pinhole_manifest.json"
        if marker.is_file():
            manifest = json.loads(marker.read_text(encoding="utf-8"))
            if manifest.get("source") == str(source_scene):
                print(f"[pinhole] Reusing validated dataset: {destination_scene}")
                return
        raise FileExistsError(
            f"Destination already exists but has no matching completed manifest: {destination_scene}. "
            "Choose a new BTS_PINHOLE_DATA_ROOT; do not overwrite a partial experiment."
        )

    cameras = read_cameras_binary(str(source_sparse / "cameras.bin"))
    unsupported = [camera.model for camera in cameras.values() if camera.model not in {"SIMPLE_RADIAL", "RADIAL"}]
    if unsupported:
        raise ValueError(f"Expected radial cameras only, found {unsupported} in {source_scene}.")
    name_to_camera = image_to_camera_ids(source_sparse)
    maps = image_camera_map(cameras)
    destination_train = output_training_root(source_scene, destination_scene)
    destination_sparse = destination_train / "sparse" / "0"

    try:
        copy_sparse_without_camera(source_sparse, destination_sparse)
        pinhole_cameras = {camera_id: pinhole_camera(camera) for camera_id, camera in cameras.items()}
        write_cameras_binary(pinhole_cameras, str(destination_sparse / "cameras.bin"))

        image_count = remap_directory(
            source_train / "images", destination_train / "images", name_to_camera, maps, jpeg_quality, False
        )
        mask_counts = {}
        for directory in MASK_DIRECTORIES:
            count = remap_directory(
                source_train / directory, destination_train / directory, name_to_camera, maps, jpeg_quality, True
            )
            if count:
                mask_counts[directory] = count
        copy_test_metadata(source_scene, destination_scene)
        for root_file in ("README.txt",):
            candidate = source_scene / root_file
            if candidate.is_file():
                shutil.copy2(candidate, destination_scene / root_file)
        manifest = {
            "source": str(source_scene),
            "camera_model": "PINHOLE",
            "image_count": image_count,
            "mask_counts": mask_counts,
            "image_size": [next(iter(cameras.values())).width, next(iter(cameras.values())).height],
            "camera_params": {str(key): pinhole_cameras[key].params.tolist() for key in pinhole_cameras},
        }
        (destination_scene / ".pinhole_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except BaseException:
        # Do not delete automatically: an unexpected exception can be useful
        # for debugging, and subsequent runs will refuse this partial output.
        raise
    print(f"[pinhole] Built {destination_scene}: {image_count} images, masks={mask_counts}")


def main() -> int:
    args = parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")
    prepare(args.source, args.destination, args.jpeg_quality)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[pinhole] ERROR: {exc}", file=sys.stderr)
        raise
