"""Phase 2: resume completed BTS scenes from their 70k pinhole checkpoints.

Attach an input dataset containing extracted or zipped archives named
``chkpnt70000_hcm0421`` through ``chkpnt70000_hcm0674``.  It preserves
the successful Phase-1 geometry, gives the tower lattice a small late density
budget, and deliberately performs no global pruning.  ``bonsai`` and
``chair`` are excluded: their floater repair must use per-test-view culls
after inspecting their renders, not destructive training-time pruning.
"""

import os


os.environ.update({
    "BTS_REQUIRE_REPO_SYNC": "1",
    "BTS_DATA_DIR": "/kaggle/input/datasets/tdukaggle/ai-race-data/data/data",
    # The durable Phase-1 archives are read-only Kaggle Input.  Do not rely
    # on a prior session's /kaggle/working directory surviving a restart.
    "BTS_CHECKPOINT_INPUT_DIR": "/kaggle/input/datasets/tdukaggle/ai-race-data/checkpoint_final",
    "BTS_SCENES": "HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    "BTS_PINHOLE_PREPROCESS_SCENES": "HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    "BTS_PINHOLE_DATA_ROOT": "/kaggle/working/data_pinhole_full_v1",
    "BTS_PINHOLE_JPEG_QUALITY": "100",
    "BTS_OUTPUT_DIR": "/kaggle/working/output_pinhole_full_v1",
    "BTS_CHECKPOINT_DIR": "/kaggle/working/checkpoints_pinhole_full_v1",
    "BTS_SUBMISSION_DIR": "/kaggle/working/submission_phase2_bts",
    "BTS_SUBMISSION_ZIP": "/kaggle/working/submission_phase2_bts.zip",

    # Resume exactly from Phase 1.  A missing 70k checkpoint is a hard error:
    # never silently start a new model in this refinement profile.
    "BTS_FRESH_RUN": "0",
    "BTS_FRESH_SCENES": "",
    "BTS_RESUME_LOCAL": "1",
    "BTS_RESUME_INPUT": "1",
    # Do not inherit the notebook's legacy generic finetune default: that
    # branch disables densification and turns on prune-only cleanup.
    "BTS_FINETUNE_SCENES": "",
    "BTS_REQUIRE_RESUME_SCENES": "HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    "BTS_REQUIRE_CHECKPOINT_ARCHIVE_SCENES": "HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    "BTS_HCM0421_REQUIRE_RESUME_MIN_ITERATION": "70000",
    "BTS_HCM0539_REQUIRE_RESUME_MIN_ITERATION": "70000",
    "BTS_HCM0540_REQUIRE_RESUME_MIN_ITERATION": "70000",
    "BTS_HCM0644_REQUIRE_RESUME_MIN_ITERATION": "70000",
    "BTS_HCM0674_REQUIRE_RESUME_MIN_ITERATION": "70000",

    "BTS_ITERATIONS": "80000",
    "BTS_HCM0421_ITERATIONS": "80000",
    "BTS_HCM0539_ITERATIONS": "80000",
    "BTS_HCM0540_ITERATIONS": "80000",
    "BTS_HCM0644_ITERATIONS": "80000",
    "BTS_HCM0674_ITERATIONS": "80000",
    "BTS_POSITION_LR_MAX_STEPS": "100000",
    "BTS_CHECKPOINT_ITERATIONS": "70000,80000",
    "BTS_VALIDATION_ITERATIONS": "70000,80000",
    "BTS_VALIDATION_HOLDOUT": "0",

    # 6.4M at 70k leaves useful room below the T4-tested 8.2M ceiling.
    # At most 800k late points can be born; no broad prune can remove thin
    # tower struts or create the black holes seen in prior experiments.
    "BTS_MAX_GAUSSIANS": "8200000",
    "BTS_DENSIFY_UNTIL_ITER": "78000",
    "BTS_DENSIFY_GRAD_THRESHOLD": "0.00008",
    "BTS_MAX_NEW_POINTS_PER_DENSIFY": "10000",
    "BTS_DISABLE_DENSIFY_PRUNE": "1",
    "BTS_DENSIFY_CAP_SCHEDULE": "70000:6800000,74000:7400000,77000:8000000",
    "BTS_PERCENT_DENSE": "0.003",
    "BTS_PRUNE_ONLY_UNTIL_ITER": "0",
    "BTS_PRUNE_ONLY_FROM_ITER": "0",
    "BTS_PRUNE_OPACITY_THRESHOLD": "0.0001",
    "BTS_PRUNE_MIN_VISIBILITY": "0",
    "BTS_MAX_SCREEN_SIZE": "0",
    "BTS_TEST_POSE_PRUNE_DISTANCE": "0",

    # Small detail emphasis, still limited to verified BTS masks.
    "BTS_FOREGROUND_LOSS_WEIGHT": "25.0",
    "BTS_FOREGROUND_EDGE_LOSS_WEIGHT": "0.12",
    "BTS_IMAGE_EDGE_LOSS_WEIGHT": "0.07",
    "BTS_REQUIRE_FOREGROUND_MASK_SCENES": "HCM0421,HCM0539,HCM0540,HCM0644,HCM0674",
    "BTS_MIN_FOREGROUND_MASK_COVERAGE": "0.40",
    "BTS_DEPTH_WEIGHT_INIT": "0.0",
    "BTS_SH_DEGREE": "2",
    "BTS_ANTIALIASING": "1",
    "BTS_KEEP_MODEL_ARTIFACTS": "1",
    "BTS_CHECKPOINT_ARCHIVE_ZIP": "1",
    "BTS_MAX_WORKERS": "2",
    "BTS_TIME_LIMIT_H": "7.5",
    "BTS_STOP_BUFFER_MIN": "25",
    "BTS_MIN_FREE_DISK_GB": "4.0",
    "BTS_DISK_CHECK_INTERVAL": "100",
    "BTS_TRAIN_RESOLUTION": "1",
    "BTS_RENDER_RESOLUTION": "1",
})

print("BTS Phase 2 loaded: resume 70k -> 80k, late densification only, no global pruning.")
