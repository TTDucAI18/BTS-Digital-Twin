"""Kaggle cleanup pass for floater removal; run before kaggle_notebook.py.

It resumes completed 40k/60k checkpoints and runs only prune/refinement steps.
No scene is retrained from scratch and no new Gaussian is cloned or split.
"""

import os

os.environ.update({
    # Matches the input layout used by the preceding Kaggle run.  Change both
    # paths only if the dataset is attached under a different Kaggle slug.
    "BTS_DATA_DIR": "/kaggle/input/datasets/tdukaggle/ai-race-data/data/data",
    "BTS_CHECKPOINT_INPUT_DIR": "/kaggle/input/datasets/tdukaggle/ai-race-data/checkpoints/checkpoints",
    "BTS_CHECKPOINT_DIR": "/kaggle/working/checkpoints",
    "BTS_OUTPUT_DIR": "/kaggle/working/output",
    "BTS_SUBMISSION_DIR": "/kaggle/working/submission",
    "BTS_SUBMISSION_ZIP": "/kaggle/working/submission.zip",
    "BTS_RESUME_LOCAL": "1",
    "BTS_RESUME_INPUT": "0",
    "BTS_FRESH_RUN": "0",
    # Critical: do not reset bonsai/chair in the cleanup pass.
    "BTS_FRESH_SCENES": "",

    # Existing final checkpoints: close-up=40k, BTS=60k.
    "BTS_ITERATIONS": "60000",
    "BTS_POSITION_LR_MAX_STEPS": "62500",
    "BTS_CLOSEUP_ITERATIONS": "40000",
    "BTS_CLEANUP_SCENES": "bonsai,chair,HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    "BTS_CLOSEUP_CLEANUP_STEPS": "4000",
    "BTS_CLEANUP_STEPS": "2500",

    # No densification is reachable from resumed iterations; these values are
    # kept at their original phase boundaries as an explicit safety guard.
    "BTS_DENSIFY_UNTIL_ITER": "50000",
    "BTS_CLOSEUP_DENSIFY_UNTIL_ITER": "30000",
    "BTS_MAX_GAUSSIANS": "8000000",
    "BTS_CLOSEUP_MAX_GAUSSIANS": "8000000",

    # Conservative general cleanup, with stronger screen-space rejection for
    # close-up floaters and HCM0421's tower occluder.
    "BTS_CLEANUP_PRUNE_OPACITY_THRESHOLD": "0.008",
    "BTS_CLEANUP_MAX_SCREEN_SIZE": "18",
    "BTS_CLOSEUP_CLEANUP_PRUNE_OPACITY_THRESHOLD": "0.010",
    "BTS_CLOSEUP_CLEANUP_MAX_SCREEN_SIZE": "24",
    "BTS_HCM0421_CLEANUP_PRUNE_OPACITY_THRESHOLD": "0.010",
    "BTS_HCM0421_CLEANUP_MAX_SCREEN_SIZE": "14",

    "BTS_CHECKPOINT_ITERATIONS": "40000,42500,44000,60000,62500,63000",
    "BTS_VALIDATION_ITERATIONS": "40000,44000,60000,62500,63000",
    "BTS_MAX_WORKERS": "2",
    "BTS_TIME_LIMIT_H": "11.5",
    "BTS_STOP_BUFFER_MIN": "30",
    "BTS_MIN_FREE_DISK_GB": "2.0",
    "BTS_KEEP_MODEL_ARTIFACTS": "0",
    "BTS_SCENES": "bonsai,chair,HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
})
