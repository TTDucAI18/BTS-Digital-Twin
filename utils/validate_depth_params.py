"""Validate Depth Anything scale/offset fits against sparse COLMAP points."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from read_write_model import qvec2rotmat, read_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--depths-dir", required=True, type=Path)
    args = parser.parse_args()
    base = args.base_dir
    params = json.loads((base / "sparse" / "0" / "depth_params.json").read_text())
    cameras, images, points3d = read_model(str(base / "sparse" / "0"), ext=".bin")
    point_ids = np.array([point.id for point in points3d.values()])
    points = np.zeros((point_ids.max() + 1, 3))
    points[point_ids] = np.array([point.xyz for point in points3d.values()])
    rows = []
    for image in images.values():
        stem = Path(image.name).stem
        depth_path = args.depths_dir / f"{stem}.png"
        if stem not in params or not depth_path.is_file():
            continue
        ids = image.point3D_ids
        valid_ids = (ids >= 0) & (ids < len(points))
        xyz = points[ids[valid_ids]]
        uv = image.xys[valid_ids].astype(np.float32)
        z = (xyz @ qvec2rotmat(image.qvec).T + image.tvec)[:, 2]
        camera = cameras[image.camera_id]
        valid = (
            np.isfinite(z) & (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0)
            & (uv[:, 0] < camera.width) & (uv[:, 1] < camera.height)
        )
        mono_map = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65536.0
        mono = cv2.remap(mono_map, uv[valid, 0], uv[valid, 1], cv2.INTER_LINEAR).reshape(-1)
        colmap = 1.0 / z[valid]
        scale, offset = params[stem]["scale"], params[stem]["offset"]
        fitted = scale * mono + offset
        corr = np.corrcoef(mono, colmap)[0, 1]
        relative_error = np.median(np.abs(fitted - colmap) / (np.abs(colmap) + 1e-6))
        rows.append((stem, len(colmap), corr, relative_error))

    values = np.array([row[1:] for row in rows], dtype=float)
    print(f"validated frames: {len(rows)}")
    print("points/frame (p1,p50,p99):", np.percentile(values[:, 0], [1, 50, 99]))
    print("Pearson correlation (p1,p10,p50,p90,p99):", np.percentile(values[:, 1], [1, 10, 50, 90, 99]))
    print("median relative fit error (p10,p50,p90):", np.percentile(values[:, 2], [10, 50, 90]))
    print("negative correlation:", int((values[:, 1] <= 0).sum()))
    print("correlation < 0.2:", int((values[:, 1] < 0.2).sum()))
    print("relative error > 50%:", int((values[:, 2] > 0.5).sum()))
    for stem, _, corr, error in sorted(rows, key=lambda row: row[2])[:10]:
        print(f"worst-corr {stem}: corr={corr:.3f}, rel-error={error:.3f}")


if __name__ == "__main__":
    main()
