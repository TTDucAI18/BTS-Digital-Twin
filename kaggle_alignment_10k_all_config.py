"""Conservative 10k fixed-geometry alignment for the final checkpoints.

Input contract:
* bonsai and all HCM scenes: trusted step-80k archive;
* chair: trusted step-70k archive.

No Gaussian is added or pruned.  The optimizer only makes small RGB, opacity,
scale, rotation, and position corrections.  Run this file/cell before
``kaggle_notebook.py``.
"""

import os


os.environ.update({
    "BTS_DATA_DIR": "/kaggle/input/datasets/tdukaggle/ai-race-data/data/data",
    # Attach one dataset containing the six 80k archives and chair's 70k
    # archive.  Archives may be extracted folders or .zip files, not raw .pth.
    "BTS_CHECKPOINT_INPUT_DIR": "/kaggle/input/ai-race-alignment-checkpoints",
    "BTS_CHECKPOINT_DIR": "/kaggle/working/checkpoints_alignment_10k_v1",
    "BTS_OUTPUT_DIR": "/kaggle/working/output_alignment_10k_v1",
    "BTS_SUBMISSION_DIR": "/kaggle/working/submission_alignment_10k_v1",
    "BTS_SUBMISSION_ZIP": "/kaggle/working/submission_alignment_10k_v1.zip",

    "BTS_SCENES": "bonsai,chair,HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    "BTS_FRESH_RUN": "0",
    "BTS_FRESH_SCENES": "",
    "BTS_FINETUNE_SCENES": "",
    "BTS_CLEANUP_SCENES": "",
    "BTS_CLEANUP_STEPS": "0",
    "BTS_CLOSEUP_CLEANUP_STEPS": "0",
    "BTS_RENDER_ONLY_SCENES": "",
    "BTS_RESUME_LOCAL": "1",
    "BTS_RESUME_INPUT": "0",

    # Refuse a wrong/missing source checkpoint instead of silently training a
    # scene from scratch.  The non-chair sources must be the current 80k set.
    "BTS_REQUIRE_RESUME_SCENES": "bonsai,chair,HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    "BTS_REQUIRE_RESUME_MIN_ITERATION": "70000",
    "BTS_BONSAI_REQUIRE_RESUME_MIN_ITERATION": "80000",
    "BTS_HCM0421_REQUIRE_RESUME_MIN_ITERATION": "80000",
    "BTS_HCM0539_REQUIRE_RESUME_MIN_ITERATION": "80000",
    "BTS_HCM0540_REQUIRE_RESUME_MIN_ITERATION": "80000",
    "BTS_HCM0644_REQUIRE_RESUME_MIN_ITERATION": "80000",
    "BTS_HCM0674_REQUIRE_RESUME_MIN_ITERATION": "80000",

    # Six scenes: 80k -> 90k.  Chair: 70k -> 80k.
    "BTS_ITERATIONS": "90000",
    "BTS_CLOSEUP_ITERATIONS": "90000",
    "BTS_CHAIR_ITERATIONS": "80000",
    "BTS_CHECKPOINT_ITERATIONS": "80000,85000,90000",
    "BTS_VALIDATION_ITERATIONS": "85000,90000",

    # Fixed geometry: iteration starts exactly at the densification boundary,
    # making both densification and every in-training pruning path inactive.
    "BTS_DENSIFY_UNTIL_ITER": "80000",
    "BTS_CLOSEUP_DENSIFY_UNTIL_ITER": "80000",
    "BTS_CHAIR_DENSIFY_UNTIL_ITER": "70000",
    "BTS_TEST_POSE_PRUNE_DISTANCE": "0",
    "BTS_MAX_SCREEN_SIZE": "0",
    "BTS_CLOSEUP_MAX_SCREEN_SIZE": "0",
    "BTS_PRUNE_OPACITY_THRESHOLD": "0.0001",
    "BTS_CLOSEUP_PRUNE_OPACITY_THRESHOLD": "0.0001",
    "BTS_DEPTH_WEIGHT_INIT": "0.0",
    "BTS_CLOSEUP_DEPTH_WEIGHT_INIT": "0.0",
    "BTS_CHAIR_DEPTH_WEIGHT_INIT": "0.0",

    # Optimizer corrections are deliberately much smaller than a normal
    # training phase.  Point count and VRAM remain effectively unchanged.
    "BTS_POSITION_LR_MAX_STEPS": "160000",
    "BTS_CLOSEUP_POSITION_LR_MAX_STEPS": "160000",
    "BTS_ALIGNMENT_POSITION_LR_SCALE": "0.30",
    "BTS_ALIGNMENT_FEATURE_LR_SCALE": "0.20",
    "BTS_ALIGNMENT_OPACITY_LR_SCALE": "0.10",
    "BTS_ALIGNMENT_SCALING_LR_SCALE": "0.10",
    "BTS_ALIGNMENT_ROTATION_LR_SCALE": "0.20",

    # Preserve the existing detail emphasis without letting depth priors pull
    # a settled reconstruction toward per-view monocular estimates.
    "BTS_FOREGROUND_LOSS_WEIGHT": "20.0",
    "BTS_FOREGROUND_EDGE_LOSS_WEIGHT": "0.08",
    "BTS_BONSAI_FOREGROUND_LOSS_WEIGHT": "8.0",
    "BTS_BONSAI_FOREGROUND_EDGE_LOSS_WEIGHT": "0.05",
    "BTS_CHAIR_FOREGROUND_LOSS_WEIGHT": "5.0",
    "BTS_CHAIR_FOREGROUND_EDGE_LOSS_WEIGHT": "0.04",
    "BTS_IMAGE_EDGE_LOSS_WEIGHT": "0.04",
    "BTS_CLOSEUP_IMAGE_EDGE_LOSS_WEIGHT": "0.03",
    "BTS_DISABLE_FOREGROUND_MASK_SCENES": "",

    # Floater policy: never delete from the model in this tail.  Add exact
    # image rules only after confirming the affected render(s), e.g.
    # "frame_000123.jpg:0.08:0.0".  A global near-camera cull would risk
    # removing real chair legs, bonsai branches, or antenna tips.
    "BTS_RENDER_NEAR_CAMERA_DISTANCE": "0.0",
    "BTS_RENDER_NEAR_CAMERA_SCALE_TO_DISTANCE": "0.0",
    "BTS_CHAIR_RENDER_NEAR_CAMERA_CULLS": "",
    "BTS_BONSAI_RENDER_NEAR_CAMERA_CULLS": "",
    "BTS_RENDER_ENSEMBLE_SCALES": "1.0",
    "BTS_CLOSEUP_RENDER_ENSEMBLE_SCALES": "1.0",
    "BTS_CLOSEUP_RENDER_SHARPEN_AMOUNT": "0.0",
    "BTS_ANTIALIASING": "1",

    # Retain close-up models for floater inspection.  HCM final archives
    # remain resumable in CHECKPOINT_DIR but their large working artifacts are
    # released after the render set is copied.
    "BTS_KEEP_MODEL_ARTIFACTS": "1",
    "BTS_KEEP_MODEL_SCENES": "bonsai,chair",
    "BTS_CHECKPOINT_BACKUP_KEEP": "2",
    "BTS_CHECKPOINT_ARCHIVE_ZIP": "1",

    "BTS_MAX_WORKERS": "2",
    "BTS_TIME_LIMIT_H": "11.5",
    "BTS_STOP_BUFFER_MIN": "30",
    "BTS_MIN_FREE_DISK_GB": "2.0",
    "BTS_DISK_CHECK_INTERVAL": "100",
    "BTS_TRAIN_RESOLUTION": "1",
    "BTS_RENDER_RESOLUTION": "1",
})
