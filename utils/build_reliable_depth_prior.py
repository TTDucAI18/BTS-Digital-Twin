"""Build a quality-gated Depth Anything prior aligned to sparse COLMAP."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from read_write_model import qvec2rotmat, read_model


def robust_affine(source, target):
    """Fit target ~= scale * source + offset with MAD-trimmed least squares."""
    keep = np.ones(len(source), dtype=bool)
    for _ in range(4):
        matrix = np.column_stack((source[keep], np.ones(keep.sum())))
        scale, offset = np.linalg.lstsq(matrix, target[keep], rcond=None)[0]
        residual = np.abs(scale * source + offset - target)
        median = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - median))
        if not np.isfinite(mad) or mad <= 1e-8:
            break
        updated = residual <= median + 3.0 * 1.4826 * mad
        if updated.sum() < 32 or np.array_equal(updated, keep):
            break
        keep = updated
    return float(scale), float(offset), keep


def main():
    parser = argparse.ArgumentParser(description="Keep only COLMAP-consistent Depth Anything maps.")
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--source-depths", required=True, type=Path)
    parser.add_argument("--out-depths", required=True, type=Path)
    parser.add_argument("--min-correlation", type=float, default=0.45)
    parser.add_argument("--max-relative-error", type=float, default=0.40)
    parser.add_argument("--min-points", type=int, default=75)
    parser.add_argument("--max-point-error", type=float, default=2.0)
    parser.add_argument("--min-track-length", type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.min_correlation <= 1 or args.max_relative_error <= 0 or args.min_points < 32:
        raise ValueError("Invalid reliability thresholds")

    cameras, images, points3d = read_model(str(args.base_dir / "sparse" / "0"), ext=".bin")
    point_ids = np.array([point.id for point in points3d.values()])
    xyz = np.zeros((point_ids.max() + 1, 3))
    errors = np.full(point_ids.max() + 1, np.inf)
    tracks = np.zeros(point_ids.max() + 1, dtype=int)
    for point in points3d.values():
        xyz[point.id] = point.xyz
        errors[point.id] = point.error
        tracks[point.id] = len(point.image_ids)

    args.out_depths.mkdir(parents=True, exist_ok=True)
    accepted, report = {}, {}
    for image in images.values():
        stem = Path(image.name).stem
        source_path = args.source_depths / f"{stem}.png"
        if not source_path.is_file():
            continue
        ids = image.point3D_ids
        safe_ids = ids.clip(min=0, max=len(xyz) - 1)
        valid_ids = (
            (ids >= 0) & (ids < len(xyz)) & (errors[safe_ids] <= args.max_point_error)
            & (tracks[safe_ids] >= args.min_track_length)
        )
        points = xyz[ids[valid_ids]]
        uv = image.xys[valid_ids].astype(np.float32)
        z = (points @ qvec2rotmat(image.qvec).T + image.tvec)[:, 2]
        camera = cameras[image.camera_id]
        valid = (
            np.isfinite(z) & (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0)
            & (uv[:, 0] < camera.width) & (uv[:, 1] < camera.height)
        )
        if valid.sum() < args.min_points:
            report[stem] = {"accepted": False, "reason": "too_few_colmap_points", "points": int(valid.sum())}
            continue
        encoded = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if encoded is None or encoded.ndim != 2 or encoded.dtype != np.uint16:
            report[stem] = {"accepted": False, "reason": "invalid_depth_map"}
            continue
        mono_map = encoded.astype(np.float32) / 65536.0
        mono = cv2.remap(mono_map, uv[valid, 0], uv[valid, 1], cv2.INTER_LINEAR).reshape(-1)
        colmap = 1.0 / z[valid]
        corr = float(np.corrcoef(mono, colmap)[0, 1])
        invert = corr < 0.0
        if invert:
            mono = 1.0 - mono
            corr = -corr
        scale, offset, inliers = robust_affine(mono, colmap)
        fitted = scale * mono + offset
        relative_error = float(np.median(np.abs(fitted - colmap) / (np.abs(colmap) + 1e-6)))
        accepted_frame = (
            np.isfinite(corr) and scale > 0 and corr >= args.min_correlation
            and relative_error <= args.max_relative_error and inliers.sum() >= args.min_points
        )
        report[stem] = {
            "accepted": bool(accepted_frame), "points": int(len(colmap)), "inliers": int(inliers.sum()),
            "correlation": corr, "relative_error": relative_error, "inverted": invert,
        }
        if not accepted_frame:
            continue
        if invert:
            encoded = (65536 - encoded.astype(np.uint32)).astype(np.uint16)
        destination = args.out_depths / source_path.name
        if not cv2.imwrite(str(destination), encoded):
            raise RuntimeError(f"Could not write {destination}")
        accepted[stem] = {"scale": scale, "offset": offset}

    params_path = args.base_dir / "sparse" / "0" / "depth_params.json"
    params_path.write_text(json.dumps(accepted, indent=2), encoding="utf-8")
    (args.base_dir / "sparse" / "0" / "depth_reliability.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"accepted {len(accepted)}/{len(report)} maps -> {args.out_depths}")


if __name__ == "__main__":
    main()
