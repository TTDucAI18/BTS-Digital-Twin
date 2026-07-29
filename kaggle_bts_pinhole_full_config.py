"""High-quality full submission after the camera ablation has selected pinhole.

The five BTS scenes are rebuilt from scratch in the same pinhole model used
by hidden test poses.  Bonsai/chair are rendered from the current trusted 80k
and 70k checkpoints, respectively; they are never pruned in this run.

Attach an input dataset containing ``chkpnt80000_bonsai`` and
``chkpnt70000_chair`` archives (zip or extracted directories), then run this
file before ``kaggle_notebook.py``.

Kaggle usage: paste this entire file into the first code cell, adjust only
the two input mount paths below, then run the current kaggle_notebook.py from
top to bottom.  The notebook now validates both archives and BTS mask coverage
before allocating a GPU.
"""

import os


os.environ.update({
    "BTS_DATA_DIR": "/kaggle/input/datasets/tdukaggle/ai-race-data/data/data",
    "BTS_CHECKPOINT_INPUT_DIR": "/kaggle/input/ai-race-best-closeup-checkpoints",
    # Refuse a stale local checkout: the notebook must pull the committed main
    # revision that contains prepare_pinhole_dataset.py before it starts.
    "BTS_REQUIRE_REPO_SYNC": "1",
    "BTS_SCENES": "bonsai,chair,HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    "BTS_PINHOLE_PREPROCESS_SCENES": "HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    "BTS_PINHOLE_DATA_ROOT": "/kaggle/working/data_pinhole_full_v1",
    "BTS_PINHOLE_JPEG_QUALITY": "100",

    "BTS_OUTPUT_DIR": "/kaggle/working/output_pinhole_full_v1",
    "BTS_CHECKPOINT_DIR": "/kaggle/working/checkpoints_pinhole_full_v1",
    "BTS_SUBMISSION_DIR": "/kaggle/working/submission_pinhole_full_v1",
    "BTS_SUBMISSION_ZIP": "/kaggle/working/submission_pinhole_full_v1.zip",

    # Fresh models for transformed HCM data.  The close-up scenes are exact
    # render-only restores, so this experiment cannot degrade their geometry.
    "BTS_FRESH_RUN": "0",
    "BTS_FRESH_RUN_ID": "pinhole-full-v1",
    "BTS_FRESH_SCENES": "HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    "BTS_RESUME_LOCAL": "1",
    "BTS_RESUME_INPUT": "1",
    "BTS_REQUIRE_RESUME_SCENES": "bonsai,chair",
    "BTS_REQUIRE_CHECKPOINT_ARCHIVE_SCENES": "bonsai,chair",
    "BTS_BONSAI_REQUIRE_RESUME_MIN_ITERATION": "80000",
    "BTS_CHAIR_REQUIRE_RESUME_MIN_ITERATION": "70000",
    "BTS_RENDER_ONLY_SCENES": "bonsai,chair",
    "BTS_FINETUNE_SCENES": "",
    "BTS_CLEANUP_SCENES": "",
    "BTS_CLEANUP_STEPS": "0",
    "BTS_CLOSEUP_CLEANUP_STEPS": "0",

    "BTS_ITERATIONS": "70000",
    "BTS_CLOSEUP_ITERATIONS": "80000",
    "BTS_BONSAI_ITERATIONS": "80000",
    "BTS_CHAIR_ITERATIONS": "70000",
    "BTS_POSITION_LR_MAX_STEPS": "100000",
    "BTS_CHECKPOINT_ITERATIONS": "35000,55000,70000",
    "BTS_VALIDATION_ITERATIONS": "55000,70000",
    # Final training uses every frame after ablation establishes the policy.
    "BTS_VALIDATION_HOLDOUT": "0",
    "BTS_VALIDATION_FRACTION": "0.05",

    # Density is conservative enough for a 16GB T4 yet gives the tower
    # lattice capacity until late training.  No screen-size or test-pose cull
    # is allowed; their previous use created holes in thin geometry.
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
    "BTS_REQUIRE_FOREGROUND_MASK_SCENES": "HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    # HCM0539 has the lowest observed mask availability (~45%); require at
    # least 40% so a wrongly mounted/no-mask dataset fails before training.
    "BTS_MIN_FOREGROUND_MASK_COVERAGE": "0.40",
    "BTS_IMAGE_EDGE_LOSS_WEIGHT": "0.05",

    # Retain the current bonsai/chair render policy: no geometry mutation and
    # no global floater cull.  Exact per-frame culls remain available later.
    "BTS_RENDER_NEAR_CAMERA_DISTANCE": "0.0",
    "BTS_RENDER_NEAR_CAMERA_SCALE_TO_DISTANCE": "0.0",
    "BTS_RENDER_ENSEMBLE_SCALES": "1.0",
    "BTS_CLOSEUP_RENDER_ENSEMBLE_SCALES": "1.0",
    "BTS_CLOSEUP_RENDER_SHARPEN_AMOUNT": "0.0",
    "BTS_ANTIALIASING": "1",

    "BTS_KEEP_MODEL_ARTIFACTS": "1",
    "BTS_KEEP_MODEL_SCENES": "bonsai,chair",
    "BTS_CHECKPOINT_ARCHIVE_ZIP": "1",
    "BTS_CHECKPOINT_BACKUP_KEEP": "2",
    "BTS_MAX_WORKERS": "2",
    "BTS_TIME_LIMIT_H": "11.7",
    "BTS_STOP_BUFFER_MIN": "35",
    "BTS_MIN_FREE_DISK_GB": "4.0",
    "BTS_DISK_CHECK_INTERVAL": "100",
    "BTS_TRAIN_RESOLUTION": "1",
    "BTS_RENDER_RESOLUTION": "1",
})

# Keep the render-only checkpoint contract explicit in the configuration cell.
# If a stale/partial cell is pasted into Kaggle, fail here rather than letting
# chair inherit BTS_CLOSEUP_ITERATIONS=80000 and fail much later in preflight.
_required_targets = {
    "BTS_BONSAI_ITERATIONS": "80000",
    "BTS_CHAIR_ITERATIONS": "70000",
}
for _name, _expected in _required_targets.items():
    if os.environ.get(_name) != _expected:
        raise RuntimeError(f"{_name} must be {_expected}, got {os.environ.get(_name)!r}")
print(
    "Pinhole full profile loaded: "
    f"bonsai={os.environ['BTS_BONSAI_ITERATIONS']}k, "
    f"chair={os.environ['BTS_CHAIR_ITERATIONS']}k, "
    "five BTS scenes=fresh 70k."
)
