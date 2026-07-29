"""Controlled camera-model ablation for the two representative BTS scenes.

Run twice, changing only ``BTS_CAMERA_ABLATION_MODE`` before this file:

* ``radial``  : current pipeline (control);
* ``pinhole`` : undistorted training images/masks plus PINHOLE cameras.

Compare the final ``Evaluating validation`` lines at 70k.  The 5% evenly
spaced holdout is never used for gradients, so it is a meaningful model
selection signal even though Kaggle test images are hidden.
"""

import os


mode = os.environ.get("BTS_CAMERA_ABLATION_MODE", "pinhole").strip().lower()
if mode not in {"radial", "pinhole"}:
    raise ValueError("BTS_CAMERA_ABLATION_MODE must be 'radial' or 'pinhole'.")

selected = "HCM0421,HCM0644"
pinhole_scenes = selected if mode == "pinhole" else ""
run_id = f"camera-{mode}-validation-v1"

os.environ.update({
    "BTS_DATA_DIR": "/kaggle/input/datasets/tdukaggle/ai-race-data/data/data",
    "BTS_SCENES": selected,
    "BTS_PINHOLE_PREPROCESS_SCENES": pinhole_scenes,
    "BTS_PINHOLE_DATA_ROOT": f"/kaggle/working/data_pinhole_{mode}_validation_v1",
    "BTS_PINHOLE_JPEG_QUALITY": "100",

    "BTS_OUTPUT_DIR": f"/kaggle/working/output_camera_{mode}_validation_v1",
    "BTS_CHECKPOINT_DIR": f"/kaggle/working/checkpoints_camera_{mode}_validation_v1",
    "BTS_SUBMISSION_DIR": f"/kaggle/working/submission_camera_{mode}_validation_v1",
    "BTS_SUBMISSION_ZIP": f"/kaggle/working/submission_camera_{mode}_validation_v1.zip",

    # Never resume a radial checkpoint into pinhole data (or conversely).
    "BTS_FRESH_RUN": "0",
    "BTS_FRESH_RUN_ID": run_id,
    "BTS_FRESH_SCENES": selected,
    "BTS_RESUME_LOCAL": "0",
    "BTS_RESUME_INPUT": "0",
    "BTS_FINETUNE_SCENES": "",
    "BTS_CLEANUP_SCENES": "",
    "BTS_RENDER_ONLY_SCENES": "",

    "BTS_ITERATIONS": "70000",
    "BTS_POSITION_LR_MAX_STEPS": "100000",
    "BTS_CHECKPOINT_ITERATIONS": "35000,55000,70000",
    "BTS_VALIDATION_ITERATIONS": "35000,55000,70000",
    "BTS_VALIDATION_HOLDOUT": "1",
    "BTS_VALIDATION_FRACTION": "0.05",
    "BTS_VALIDATION_LPIPS_FINAL": "1",

    # The control and treatment differ only in camera consistency.  This is
    # the measured-safe recovery density policy, not the old prune-only tail.
    "BTS_MAX_GAUSSIANS": "8200000",
    "BTS_SH_DEGREE": "2",
    "BTS_DENSIFY_UNTIL_ITER": "62000",
    "BTS_DENSIFY_GRAD_THRESHOLD": "0.00010",
    "BTS_MAX_NEW_POINTS_PER_DENSIFY": "10000",
    "BTS_DENSIFY_CAP_SCHEDULE": "10000:1200000,17000:3200000,21000:5200000",
    "BTS_PERCENT_DENSE": "0.003",
    "BTS_PRUNE_OPACITY_THRESHOLD": "0.0005",
    "BTS_MAX_SCREEN_SIZE": "0",
    "BTS_TEST_POSE_PRUNE_DISTANCE": "0",
    "BTS_DEPTH_WEIGHT_INIT": "0.0",
    "BTS_FOREGROUND_LOSS_WEIGHT": "20.0",
    "BTS_FOREGROUND_EDGE_LOSS_WEIGHT": "0.10",
    "BTS_IMAGE_EDGE_LOSS_WEIGHT": "0.05",

    "BTS_TRAIN_RESOLUTION": "1",
    "BTS_RENDER_RESOLUTION": "1",
    "BTS_ANTIALIASING": "1",
    "BTS_RENDER_ENSEMBLE_SCALES": "1.0",
    "BTS_RENDER_NEAR_CAMERA_DISTANCE": "0.0",
    "BTS_RENDER_NEAR_CAMERA_SCALE_TO_DISTANCE": "0.0",
    "BTS_MAX_WORKERS": "2",
    "BTS_TIME_LIMIT_H": "8.5",
    "BTS_STOP_BUFFER_MIN": "30",
    "BTS_MIN_FREE_DISK_GB": "4.0",
    "BTS_CHECKPOINT_ARCHIVE_ZIP": "1",
    "BTS_CHECKPOINT_BACKUP_KEEP": "2",
})
