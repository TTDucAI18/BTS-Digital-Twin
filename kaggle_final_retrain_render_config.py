"""Kaggle profile: resume every scene from the attached checkpoint to 70k.

Run this file/cell before ``kaggle_notebook.py``.

All scenes first hydrate their latest compatible input checkpoint (currently
50k), then preserve local checkpoints for interruption-safe continuation.
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
    # Hydrate the attached input checkpoint at the same step as any stale
    # local archive; a strictly newer local checkpoint wins after an
    # interruption instead of losing progress.
    "BTS_FRESH_RUN": "0",
    "BTS_FRESH_RUN_ID": "resume-input-70k-v1",
    "BTS_FRESH_SCENES": "",
    "BTS_TRAIN_FIRST_SCENES": "",
    "BTS_RENDER_ONLY_SCENES": "",
    "BTS_RESUME_LOCAL": "1",
    "BTS_RESUME_INPUT": "1",
    "BTS_CLEANUP_SCENES": "",
    "BTS_CLEANUP_STEPS": "0",
    "BTS_CLOSEUP_CLEANUP_STEPS": "0",

    # Resume all current 50k models through a 20k refinement tail.  Continue
    # densification only through 60k, then let 60k--70k converge geometry.
    "BTS_ITERATIONS": "70000",
    "BTS_POSITION_LR_MAX_STEPS": "70000",
    "BTS_CLOSEUP_ITERATIONS": "70000",
    "BTS_CLOSEUP_POSITION_LR_MAX_STEPS": "70000",
    "BTS_CHECKPOINT_ITERATIONS": "60000,65000,70000",
    "BTS_VALIDATION_ITERATIONS": "60000,65000,70000",

    # Base BTS settings retain the existing memory-safe schedule while its
    # densification window is extended for resumed refinement.
    "BTS_MAX_GAUSSIANS": "8000000",
    "BTS_SH_DEGREE": "2",
    "BTS_MAX_NEW_POINTS_PER_DENSIFY": "75000",
    "BTS_DENSIFY_GRAD_THRESHOLD": "0.00015",
    "BTS_DENSIFY_UNTIL_ITER": "60000",
    "BTS_DENSIFY_CAP_SCHEDULE": "10000:1200000,17000:3200000,21000:5200000",
    "BTS_PERCENT_DENSE": "0.005",
    "BTS_MAX_SCREEN_SIZE": "20",
    "BTS_OPACITY_RESET_UNTIL_ITER": "15000",
    "BTS_DEPTH_WEIGHT_INIT": "0.02",
    "BTS_IMAGE_EDGE_LOSS_WEIGHT": "0.02",

    # Close-up geometry densifies to 60k, then converges to 70k.  Chair has
    # the weakest data, so its lower cap avoids fitting view-local noise.
    "BTS_CLOSEUP_MAX_GAUSSIANS": "4800000",
    "BTS_CLOSEUP_DENSIFY_GRAD_THRESHOLD": "0.00010",
    "BTS_CLOSEUP_DENSIFY_UNTIL_ITER": "60000",
    "BTS_CLOSEUP_DENSIFY_CAP_SCHEDULE": "10000:1200000,20000:2600000,30000:4000000,40000:4800000",
    "BTS_CLOSEUP_MAX_NEW_POINTS_PER_DENSIFY": "50000",
    "BTS_CLOSEUP_CLONE_BEFORE_SPLIT": "1",
    "BTS_CLOSEUP_PERCENT_DENSE": "0.01",
    "BTS_CLOSEUP_MAX_SCREEN_SIZE": "64",
    "BTS_CLOSEUP_OPACITY_RESET_UNTIL_ITER": "12000",
    "BTS_CLOSEUP_DEPTH_WEIGHT_INIT": "0.01",
    "BTS_CLOSEUP_IMAGE_EDGE_LOSS_WEIGHT": "0.03",
    "BTS_CHAIR_MAX_GAUSSIANS": "3500000",
    "BTS_CHAIR_DENSIFY_CAP_SCHEDULE": "10000:1200000,20000:2200000,40000:3000000,60000:3500000",
    # Chair masks often include a second chair or a fragment of the target.
    # Disable them only for chair; bonsai continues using its foreground masks.
    "BTS_DISABLE_FOREGROUND_MASK_SCENES": "chair",
    "BTS_FOREGROUND_LOSS_WEIGHT": "12.0",
    "BTS_FOREGROUND_EDGE_LOSS_WEIGHT": "0.05",

    # Remove splats that densification creates inside hidden test-camera
    # space.  The implementation uses a bounded point-by-camera block, so it
    # remains VRAM-safe for multi-million-Gaussian models.
    "BTS_TEST_POSE_PRUNE_DISTANCE": "0.10",

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
    "BTS_CLOSEUP_RENDER_ENSEMBLE_SCALES": "1.25",
    # Test-pose-only floater cull and a mild post-render detail recovery.
    # Values are conservative in native COLMAP units; set the radius to 0 to
    # disable this ablation without changing the trained model.
    "BTS_RENDER_NEAR_CAMERA_DISTANCE": "0.10",
    "BTS_RENDER_NEAR_CAMERA_SCALE_TO_DISTANCE": "0.0",
    "BTS_CLOSEUP_RENDER_SHARPEN_AMOUNT": "0.12",
})
