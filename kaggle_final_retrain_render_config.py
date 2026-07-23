"""Final Kaggle profile: retrain close-ups, then render fixed BTS models.

Run this file/cell before ``kaggle_notebook.py``.  It has two strict phases:

1. Fresh-train ``bonsai`` and ``chair`` to 50k iterations.
2. Only after both complete, hydrate the five BTS checkpoints at 60k and
   render them.  Those five scenes are render-only and can never retrain.

The notebook writes a ``.fresh_run_started`` marker after the initial reset,
so rerunning this exact profile after an interruption resumes bonsai/chair
instead of deleting their new checkpoint.
"""

import os


os.environ.update({
    # Kaggle dataset layout.  Update only these paths if the attached dataset
    # is published under another slug.
    "BTS_DATA_DIR": "/kaggle/input/datasets/tdukaggle/ai-race-data/data/data",
    "BTS_CHECKPOINT_INPUT_DIR": "/kaggle/input/datasets/tdukaggle/ai-race-data/checkpoints/checkpoints",
    "BTS_CHECKPOINT_DIR": "/kaggle/working/checkpoints",
    "BTS_OUTPUT_DIR": "/kaggle/working/output",
    "BTS_SUBMISSION_DIR": "/kaggle/working/submission",
    "BTS_SUBMISSION_ZIP": "/kaggle/working/submission.zip",

    # Scene selection and phase ordering.
    "BTS_SCENES": "bonsai,chair,HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    "BTS_FRESH_RUN": "0",
    "BTS_FRESH_SCENES": "bonsai,chair",
    "BTS_TRAIN_FIRST_SCENES": "bonsai,chair",
    "BTS_RENDER_ONLY_SCENES": "HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    "BTS_RESUME_LOCAL": "1",
    "BTS_RESUME_INPUT": "0",
    "BTS_CLEANUP_SCENES": "",
    "BTS_CLEANUP_STEPS": "0",
    "BTS_CLOSEUP_CLEANUP_STEPS": "0",

    # Target iterations.  HCM scenes must supply exact 60k archives; the
    # close-up pair uses its own fresh 50k schedule.
    "BTS_ITERATIONS": "60000",
    "BTS_POSITION_LR_MAX_STEPS": "60000",
    "BTS_CLOSEUP_ITERATIONS": "50000",
    "BTS_CLOSEUP_POSITION_LR_MAX_STEPS": "50000",
    "BTS_CHECKPOINT_ITERATIONS": "40000,45000,50000,60000",
    "BTS_VALIDATION_ITERATIONS": "40000,45000,50000,60000",

    # Base BTS settings.  They are not used to optimise the five render-only
    # scenes, but retain explicit, safe values for a controlled recovery run.
    "BTS_MAX_GAUSSIANS": "8000000",
    "BTS_SH_DEGREE": "2",
    "BTS_MAX_NEW_POINTS_PER_DENSIFY": "75000",
    "BTS_DENSIFY_GRAD_THRESHOLD": "0.00015",
    "BTS_DENSIFY_UNTIL_ITER": "50000",
    "BTS_DENSIFY_CAP_SCHEDULE": "10000:1200000,17000:3200000,21000:5200000",
    "BTS_PERCENT_DENSE": "0.005",
    "BTS_MAX_SCREEN_SIZE": "20",
    "BTS_OPACITY_RESET_UNTIL_ITER": "15000",
    "BTS_DEPTH_WEIGHT_INIT": "0.02",
    "BTS_IMAGE_EDGE_LOSS_WEIGHT": "0.02",

    # Close-up geometry: extend allocation through 40k, followed by a 10k
    # fixed-geometry convergence phase.  The 4.8M staged cap and 0.00010
    # gradient gate favour real multi-view detail over noise-driven floaters.
    "BTS_CLOSEUP_MAX_GAUSSIANS": "4800000",
    "BTS_CLOSEUP_DENSIFY_GRAD_THRESHOLD": "0.00010",
    "BTS_CLOSEUP_DENSIFY_UNTIL_ITER": "40000",
    "BTS_CLOSEUP_DENSIFY_CAP_SCHEDULE": "10000:1200000,20000:2600000,30000:4000000,40000:4800000",
    "BTS_CLOSEUP_MAX_NEW_POINTS_PER_DENSIFY": "50000",
    "BTS_CLOSEUP_CLONE_BEFORE_SPLIT": "1",
    "BTS_CLOSEUP_PERCENT_DENSE": "0.01",
    "BTS_CLOSEUP_MAX_SCREEN_SIZE": "64",
    "BTS_CLOSEUP_OPACITY_RESET_UNTIL_ITER": "12000",
    "BTS_CLOSEUP_DEPTH_WEIGHT_INIT": "0.01",
    "BTS_CLOSEUP_IMAGE_EDGE_LOSS_WEIGHT": "0.03",
    "BTS_FOREGROUND_LOSS_WEIGHT": "12.0",
    "BTS_FOREGROUND_EDGE_LOSS_WEIGHT": "0.05",

    # Render-only repair for the known HCM0421 near-camera floater.  Format:
    # image_name:near_distance:maximum_scale_to_distance.
    "BTS_HCM0421_RENDER_NEAR_CAMERA_CULLS": "DJI_20241230093734_0186_V.JPG:1.5:0.20",

    # Preserve final close-up artifacts for inspection; remove temporary
    # HCM models after render so the sequential render phase stays disk-safe.
    "BTS_KEEP_MODEL_ARTIFACTS": "1",
    "BTS_KEEP_MODEL_SCENES": "bonsai,chair",
    "BTS_CHECKPOINT_BACKUP_KEEP": "2",
    "BTS_CHECKPOINT_ARCHIVE_ZIP": "1",

    # Runtime and render safeguards.
    "BTS_MAX_WORKERS": "2",
    "BTS_TIME_LIMIT_H": "11.5",
    "BTS_STOP_BUFFER_MIN": "30",
    "BTS_MIN_FREE_DISK_GB": "2.0",
    "BTS_DISK_CHECK_INTERVAL": "100",
    "BTS_TRAIN_RESOLUTION": "1",
    "BTS_RENDER_RESOLUTION": "1",
    "BTS_ANTIALIASING": "1",
    "BTS_RENDER_ENSEMBLE_SCALES": "1.0",
    "BTS_CLOSEUP_RENDER_ENSEMBLE_SCALES": "1.0",
})
