"""One Kaggle profile for the post-70k recovery experiment.

Six scenes (bonsai plus the five BTS scenes) resume only their verified 70k
checkpoint and receive a small, geometry-creating 70k--78k tail.  Chair has
no trustworthy recovery checkpoint, so it starts clean and stops at 70k: no
post-training cleanup is allowed in either path.

Run this file/cell before ``kaggle_notebook.py``.
"""

import os


os.environ.update({
    # Attach a Kaggle dataset containing ONLY the six trusted 70k checkpoint
    # archives.  Do not place a 80k checkpoint in this directory: the runner
    # deliberately resumes the newest compatible artifact.
    "BTS_DATA_DIR": "/kaggle/input/datasets/tdukaggle/ai-race-data/data/data",
    "BTS_CHECKPOINT_INPUT_DIR": "/kaggle/input/ai-race-70k-checkpoints",
    # Isolate this experiment from previous 80k output/checkpoint directories.
    "BTS_CHECKPOINT_DIR": "/kaggle/working/checkpoints_detail_recovery_v1",
    "BTS_OUTPUT_DIR": "/kaggle/working/output_detail_recovery_v1",
    "BTS_SUBMISSION_DIR": "/kaggle/working/submission_detail_recovery_v1",
    "BTS_SUBMISSION_ZIP": "/kaggle/working/submission_detail_recovery_v1.zip",

    "BTS_SCENES": "bonsai,chair,HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    "BTS_FRESH_RUN": "0",
    "BTS_FRESH_RUN_ID": "chair-fresh-no-cleanup-v1",
    "BTS_FRESH_SCENES": "chair",
    # Do not enter the old 70k--80k prune-only path.
    "BTS_FINETUNE_SCENES": "",
    # Fail loudly rather than silently rebuilding a tower from scratch if a
    # 70k archive was mounted under the wrong name/path.
    "BTS_REQUIRE_RESUME_SCENES": "bonsai,HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    "BTS_REQUIRE_RESUME_MIN_ITERATION": "70000",
    "BTS_CLEANUP_SCENES": "",
    "BTS_CLEANUP_STEPS": "0",
    "BTS_CLOSEUP_CLEANUP_STEPS": "0",
    "BTS_RENDER_ONLY_SCENES": "",
    # Resume the six non-chair scenes from the isolated 70k archive directory;
    # skip generic input discovery, which may find an unwanted 80k artifact.
    "BTS_RESUME_LOCAL": "1",
    "BTS_RESUME_INPUT": "0",

    # Chair is a clean 70k reconstruction.  Its per-scene densification
    # override below leaves the final 10k as a convergence interval.
    "BTS_CHAIR_FRESH_ITERATIONS": "70000",
    "BTS_TRAIN_FIRST_SCENES": "chair",
    "BTS_TRAIN_FIRST_EXCLUSIVE": "0",

    # The six recovered scenes target 80k, densify through 78k, and then
    # receive a short fixed-geometry convergence interval.
    "BTS_ITERATIONS": "80000",
    "BTS_CLOSEUP_ITERATIONS": "80000",
    "BTS_POSITION_LR_MAX_STEPS": "140000",
    "BTS_CLOSEUP_POSITION_LR_MAX_STEPS": "140000",
    "BTS_CHECKPOINT_ITERATIONS": "70000,74000,78000,80000",
    "BTS_VALIDATION_ITERATIONS": "74000,78000,80000",

    # BTS detail recovery.  HCM0421 is already ~7.52M points at 70k; 8.2M is
    # a measured-T4-safe, small expansion rather than an aggressive 9M+ jump.
    "BTS_MAX_GAUSSIANS": "8200000",
    "BTS_SH_DEGREE": "2",
    # At 7.5--7.6M existing points, 30k/event would exhaust the 8.2M cap in
    # about 2k steps.  10k/event lets late tower gradients receive capacity.
    "BTS_MAX_NEW_POINTS_PER_DENSIFY": "10000",
    "BTS_DENSIFY_GRAD_THRESHOLD": "0.00010",
    "BTS_DENSIFY_UNTIL_ITER": "78000",
    "BTS_DENSIFY_CAP_SCHEDULE": "10000:1200000,17000:3200000,21000:5200000",
    "BTS_PERCENT_DENSE": "0.003",
    # Keep existing low-opacity/thin splats during recovery.  Floaters are
    # evaluated separately at render time, not deleted from the trained model.
    "BTS_PRUNE_OPACITY_THRESHOLD": "0.0005",
    "BTS_MAX_SCREEN_SIZE": "0",
    "BTS_TEST_POSE_PRUNE_DISTANCE": "0",
    "BTS_DEPTH_WEIGHT_INIT": "0.0",
    "BTS_IMAGE_EDGE_LOSS_WEIGHT": "0.05",

    # Correct BTS masks cover only a small image fraction, so raise their
    # relative training signal without disabling full-image RGB supervision.
    "BTS_FOREGROUND_LOSS_WEIGHT": "20.0",
    "BTS_FOREGROUND_EDGE_LOSS_WEIGHT": "0.10",
    "BTS_DISABLE_FOREGROUND_MASK_SCENES": "",

    # Bonsai resumes its ~6.0M-point 70k checkpoint.  A 6.3M cap stays inside
    # the observed T4 headroom while leaving room for true high-gradient detail.
    "BTS_BONSAI_DENSIFY_UNTIL_ITER": "78000",
    "BTS_BONSAI_MAX_GAUSSIANS": "6300000",
    "BTS_BONSAI_MAX_NEW_POINTS_PER_DENSIFY": "5000",
    "BTS_BONSAI_DENSIFY_GRAD_THRESHOLD": "0.00010",
    "BTS_BONSAI_PERCENT_DENSE": "0.003",
    "BTS_BONSAI_PRUNE_OPACITY_THRESHOLD": "0.0005",
    "BTS_BONSAI_MAX_SCREEN_SIZE": "0",
    "BTS_BONSAI_DEPTH_WEIGHT_INIT": "0.0",
    "BTS_BONSAI_IMAGE_EDGE_LOSS_WEIGHT": "0.05",
    # Bonsai occupies more pixels than a BTS tower; use an intermediate mask
    # emphasis instead of inheriting the BTS-specific value of 20.
    "BTS_BONSAI_FOREGROUND_LOSS_WEIGHT": "8.0",
    "BTS_BONSAI_FOREGROUND_EDGE_LOSS_WEIGHT": "0.05",

    # Chair starts fresh: retain the normal 60k geometry / 10k convergence
    # schedule, but remove the destructive screen-size and test-pose culls.
    "BTS_CHAIR_DENSIFY_UNTIL_ITER": "60000",
    "BTS_CHAIR_MAX_GAUSSIANS": "5000000",
    "BTS_CHAIR_MAX_NEW_POINTS_PER_DENSIFY": "30000",
    "BTS_CHAIR_DENSIFY_GRAD_THRESHOLD": "0.00010",
    "BTS_CHAIR_PERCENT_DENSE": "0.008",
    "BTS_CHAIR_PRUNE_OPACITY_THRESHOLD": "0.001",
    "BTS_CHAIR_MAX_SCREEN_SIZE": "0",
    "BTS_CHAIR_DEPTH_WEIGHT_INIT": "0.0",
    "BTS_CHAIR_IMAGE_EDGE_LOSS_WEIGHT": "0.03",
    # Chair occupies much more of its image than a BTS tower, so keep its
    # foreground weighting moderate via the per-scene loss overrides below.
    "BTS_CHAIR_FOREGROUND_LOSS_WEIGHT": "5.0",
    "BTS_CHAIR_FOREGROUND_EDGE_LOSS_WEIGHT": "0.04",

    # Keep close-up artifacts for visual inspection.  Successful HCM outputs
    # are removed after submission copy; their final portable checkpoint zip
    # remains in CHECKPOINT_DIR and avoids filling Kaggle working disk.
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
    "BTS_ANTIALIASING": "1",
    "BTS_RENDER_ENSEMBLE_SCALES": "1.0",
    # Render natively while judging geometry: 1.25x followed by bicubic
    # downsampling smooths exactly the thin detail this experiment targets.
    "BTS_CLOSEUP_RENDER_ENSEMBLE_SCALES": "1.0",
    "BTS_CLOSEUP_RENDER_SHARPEN_AMOUNT": "0.0",
    # Test-pose camera centres do not prove a surrounding sphere is empty.
    # Disable global inference culling for this geometry comparison.
    "BTS_RENDER_NEAR_CAMERA_DISTANCE": "0.0",
    "BTS_RENDER_NEAR_CAMERA_SCALE_TO_DISTANCE": "0.0",
})
