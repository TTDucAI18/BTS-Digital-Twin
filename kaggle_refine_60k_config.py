"""Kaggle configuration: resume five BTS scenes at 40k; retrain bonsai/chair.

Run this cell/script before kaggle_notebook.py.  Change only the checkpoint
input path when publishing the Kaggle dataset containing the five 40k archives.
After bonsai/chair have begun their clean run, set BTS_FRESH_SCENES to "" before
restarting an interrupted Kaggle session so they resume instead of being reset.
"""

import os

os.environ.update({
    # Data/model paths.  The checkpoint input must contain per-scene 40k
    # archives named as produced by kaggle_notebook.py (chkpnt40000_<scene>).
    "BTS_DATA_DIR": "/kaggle/input/ai-race-data/phase1",
    "BTS_CHECKPOINT_INPUT_DIR": "/kaggle/input/bts-40k-checkpoints",
    "BTS_CHECKPOINT_DIR": "/kaggle/working/checkpoints",
    "BTS_OUTPUT_DIR": "/kaggle/working/output",
    "BTS_SUBMISSION_DIR": "/kaggle/working/submission",
    "BTS_SUBMISSION_ZIP": "/kaggle/working/submission.zip",

    # Schedule: BTS resumes 40k -> 60k; close-up scenes train 0 -> 40k.
    "BTS_ITERATIONS": "60000",
    "BTS_POSITION_LR_MAX_STEPS": "60000",
    "BTS_CLOSEUP_ITERATIONS": "40000",
    "BTS_CLOSEUP_POSITION_LR_MAX_STEPS": "40000",
    "BTS_CHECKPOINT_ITERATIONS": "10000,20000,30000,35000,40000,45000,50000,55000,60000",
    "BTS_VALIDATION_ITERATIONS": "40000,60000",

    # Shared T4-safe Gaussian budget and BTS refinement densification.
    "BTS_MAX_GAUSSIANS": "8000000",
    "BTS_SH_DEGREE": "2",
    "BTS_MAX_NEW_POINTS_PER_DENSIFY": "75000",
    "BTS_DENSIFY_GRAD_THRESHOLD": "0.00015",
    "BTS_DENSIFY_UNTIL_ITER": "50000",
    "BTS_DENSIFY_CAP_SCHEDULE": "10000:1200000,17000:3200000,21000:5200000",
    "BTS_PERCENT_DENSE": "0.005",
    "BTS_MAX_SCREEN_SIZE": "20",
    "BTS_OPACITY_RESET_UNTIL_ITER": "15000",

    # Close-up allocation: 4M available by 30k, with clone-before-split to
    # prevent the prior <1M plateau.  It targets >=3M, subject to valid image
    # gradients and pruning rather than padding low-quality Gaussians.
    "BTS_CLOSEUP_MAX_GAUSSIANS": "8000000",
    "BTS_CLOSEUP_DENSIFY_GRAD_THRESHOLD": "0.00008",
    "BTS_CLOSEUP_DENSIFY_UNTIL_ITER": "30000",
    "BTS_CLOSEUP_DENSIFY_CAP_SCHEDULE": "10000:1200000,17000:2200000,30000:4000000",
    "BTS_CLOSEUP_PERCENT_DENSE": "0.01",
    "BTS_CLOSEUP_MAX_NEW_POINTS_PER_DENSIFY": "75000",
    "BTS_CLOSEUP_MAX_SCREEN_SIZE": "64",
    "BTS_CLOSEUP_OPACITY_RESET_UNTIL_ITER": "12000",
    "BTS_CLOSEUP_CLONE_BEFORE_SPLIT": "1",

    # Resume only verified local/input archives for BTS.  Freshness is scoped
    # to close-up scenes, preserving the five BTS 40k checkpoints.
    "BTS_RESUME_LOCAL": "1",
    "BTS_RESUME_INPUT": "0",
    "BTS_FRESH_RUN": "0",
    "BTS_FRESH_SCENES": "bonsai,chair",
    "BTS_KEEP_MODEL_ARTIFACTS": "0",
    "BTS_CHECKPOINT_BACKUP_KEEP": "2",
    "BTS_CHECKPOINT_ARCHIVE_ZIP": "1",

    # Runtime safeguards.
    "BTS_TIME_LIMIT_H": "11.5",
    "BTS_STOP_BUFFER_MIN": "30",
    "BTS_MIN_FREE_DISK_GB": "2.0",
    "BTS_DISK_CHECK_INTERVAL": "100",
    "BTS_MAX_WORKERS": "2",
    "BTS_TRAIN_RESOLUTION": "1",
    "BTS_RENDER_RESOLUTION": "1",
    "BTS_ANTIALIASING": "1",
    "BTS_SCENES": "bonsai,chair,HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
})
