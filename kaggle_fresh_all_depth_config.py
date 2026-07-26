"""Fresh-train all seven scenes with quality-gated Depth Anything priors.

Run this profile before ``kaggle_notebook.py``.  The Kaggle input must include
each scene's ``train/depths_any_reliable`` directory plus the corresponding
``train/sparse/0/depth_params.json`` generated locally.
"""

import os


os.environ.update({
    "BTS_DATA_DIR": "/kaggle/input/datasets/tdukaggle/ai-race-data/data/data",
    "BTS_CHECKPOINT_DIR": "/kaggle/working/checkpoints",
    "BTS_OUTPUT_DIR": "/kaggle/working/output",
    "BTS_SUBMISSION_DIR": "/kaggle/working/submission",
    "BTS_SUBMISSION_ZIP": "/kaggle/working/submission.zip",

    # Every scene starts clean exactly once under this experiment marker.
    "BTS_SCENES": "bonsai,chair,HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    "BTS_FRESH_RUN": "0",
    "BTS_FRESH_RUN_ID": "fresh-all-depth-v1",
    "BTS_FRESH_SCENES": "bonsai,chair,HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    "BTS_TRAIN_FIRST_SCENES": "",
    "BTS_RENDER_ONLY_SCENES": "",
    "BTS_RESUME_LOCAL": "1",
    "BTS_RESUME_INPUT": "0",

    # Fresh target schedule.  Checkpoints support resume if the session limit
    # is reached; no scene silently hydrates a prior model.
    "BTS_ITERATIONS": "45000",
    "BTS_POSITION_LR_MAX_STEPS": "45000",
    "BTS_CLOSEUP_ITERATIONS": "45000",
    "BTS_CLOSEUP_POSITION_LR_MAX_STEPS": "45000",
    "BTS_CHECKPOINT_ITERATIONS": "15000,30000,45000",
    "BTS_VALIDATION_ITERATIONS": "30000,45000",

    # Conservative point budgets keep full-resolution fresh training stable.
    "BTS_MAX_GAUSSIANS": "4000000",
    "BTS_DENSIFY_GRAD_THRESHOLD": "0.00015",
    "BTS_DENSIFY_UNTIL_ITER": "35000",
    "BTS_DENSIFY_CAP_SCHEDULE": "10000:1000000,20000:2400000,30000:4000000",
    "BTS_MAX_NEW_POINTS_PER_DENSIFY": "50000",
    "BTS_PERCENT_DENSE": "0.005",
    "BTS_MAX_SCREEN_SIZE": "32",
    "BTS_OPACITY_RESET_UNTIL_ITER": "12000",
    "BTS_IMAGE_EDGE_LOSS_WEIGHT": "0.02",

    "BTS_CLOSEUP_MAX_GAUSSIANS": "4000000",
    "BTS_CLOSEUP_DENSIFY_GRAD_THRESHOLD": "0.00010",
    "BTS_CLOSEUP_DENSIFY_UNTIL_ITER": "35000",
    "BTS_CLOSEUP_DENSIFY_CAP_SCHEDULE": "10000:1000000,20000:2400000,30000:4000000",
    "BTS_CLOSEUP_MAX_NEW_POINTS_PER_DENSIFY": "50000",
    "BTS_CLOSEUP_CLONE_BEFORE_SPLIT": "1",
    "BTS_CLOSEUP_PERCENT_DENSE": "0.01",
    "BTS_CLOSEUP_MAX_SCREEN_SIZE": "64",
    "BTS_CLOSEUP_OPACITY_RESET_UNTIL_ITER": "12000",
    "BTS_CLOSEUP_IMAGE_EDGE_LOSS_WEIGHT": "0.03",

    # The notebook uses only maps with finite calibrated params and at least
    # 10% scene coverage.  Chair stays mask-free; bonsai/HCM masks are kept.
    "BTS_MIN_DEPTH_COVERAGE": "0.10",
    "BTS_DEPTH_WEIGHT_INIT": "0.005",
    "BTS_CLOSEUP_DEPTH_WEIGHT_INIT": "0.005",
    "BTS_DISABLE_FOREGROUND_MASK_SCENES": "chair",
    "BTS_FOREGROUND_LOSS_WEIGHT": "12.0",
    "BTS_FOREGROUND_EDGE_LOSS_WEIGHT": "0.05",

    # Keep only the independently archived final checkpoint for resume.  Full
    # point-cloud/model artifacts for seven 4M-point scenes would exhaust the
    # Kaggle working disk before the queue finishes.
    "BTS_KEEP_MODEL_ARTIFACTS": "0",
    "BTS_KEEP_MODEL_SCENES": "",
    "BTS_CHECKPOINT_BACKUP_KEEP": "1",
    "BTS_CHECKPOINT_ARCHIVE_ZIP": "1",
    "BTS_MAX_WORKERS": "2",
    "BTS_TIME_LIMIT_H": "11.5",
    "BTS_STOP_BUFFER_MIN": "35",
    "BTS_MIN_FREE_DISK_GB": "3.0",
    "BTS_TRAIN_RESOLUTION": "1",
    "BTS_RENDER_RESOLUTION": "1",
    "BTS_ANTIALIASING": "1",
    "BTS_RENDER_ENSEMBLE_SCALES": "1.0",
    "BTS_CLOSEUP_RENDER_ENSEMBLE_SCALES": "1.25",
    "BTS_RENDER_NEAR_CAMERA_DISTANCE": "0.10",
    "BTS_RENDER_NEAR_CAMERA_SCALE_TO_DISTANCE": "0.0",
    "BTS_CLOSEUP_RENDER_SHARPEN_AMOUNT": "0.12",
})
