# =============================================================================
# BTS Digital Twin - Kaggle notebook script
#
# Copy this file into Kaggle as code cells, or run it as one Python script.
# The pipeline is tuned for Kaggle 2x T4, 16 GB VRAM per GPU:
#   1. clone/update repo and submodules
#   2. install CUDA extensions once
#   3. discover the seven competition scenes
#   4. train one scene per GPU
#   5. render test poses at full output resolution
#   6. package submission.zip
# =============================================================================

import glob
import io
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # tqdm installed later; fall back to identity wrapper
    def _tqdm(it, **kw):  # type: ignore
        return it


def _tqdm_bar(iterable, desc, total=None, unit="it"):
    """Wrap any iterable with a tqdm bar; safe before tqdm is installed."""
    return _tqdm(iterable, desc=desc, total=total or len(list(iterable)) if hasattr(iterable, '__len__') else None,
                 unit=unit, dynamic_ncols=True, leave=True)


# =============================================================================
# CELL 1 - Configuration
# =============================================================================

REPO_URL = os.environ.get("BTS_REPO_URL", "https://github.com/TTDucAI18/BTS-Digital-Twin.git")
REPO_REF = os.environ.get("BTS_REPO_REF", "main").strip()
REPO_DIR = Path(os.environ.get("BTS_REPO_DIR", "/kaggle/working/BTS-Digital-Twin"))
OUTPUT_DIR = Path(os.environ.get("BTS_OUTPUT_DIR", "/kaggle/working/output"))
# This directory is deliberately outside each scene's model directory.  A
# completed checkpoint survives cleanup of model artifacts and is easy to add
# to a Kaggle output dataset after an interrupted session.
CHECKPOINT_DIR = Path(os.environ.get("BTS_CHECKPOINT_DIR", "/kaggle/working/checkpoints"))
# Optional read-only Kaggle Input containing extracted checkpoint folders.  It
# is intentionally separate from CHECKPOINT_DIR: /kaggle/input is read-only,
# while new 40k checkpoints must be published under /kaggle/working.
CHECKPOINT_INPUT_DIR = Path(os.environ.get("BTS_CHECKPOINT_INPUT_DIR", str(CHECKPOINT_DIR)))
SUBMISSION_DIR = Path(os.environ.get("BTS_SUBMISSION_DIR", "/kaggle/working/submission"))
SUBMISSION_ZIP = Path(os.environ.get("BTS_SUBMISSION_ZIP", "/kaggle/working/submission.zip"))

DATA_ROOT_CANDIDATES = [
    Path(os.environ["BTS_DATA_DIR"]) if os.environ.get("BTS_DATA_DIR") else None,
    # Current local layout: D:/ai_race_2026/data/<scene>/{train,test}.
    Path("/kaggle/input/datasets/tdukaggle/ai-race-data"),
    Path("/kaggle/input/datasets/tdukaggle/ai-race-data/phase1"),
    Path("/kaggle/input/bts-digital-twin-phase1/phase1"),
    Path("/kaggle/input/ai-race-data/phase1"),
    Path("/kaggle/input/ai-race-data"),
    Path("D:/ai_race_2026/data"),
    Path("D:/ai_race_2026/data/phase1"),
]

# BTS scenes already have useful 40k checkpoints; give their refinement phase
# enough room for new geometry.  Close-up scenes intentionally start clean and
# retain the shorter 40k schedule below.
ITERATIONS = int(os.environ.get("BTS_ITERATIONS", "60000"))
CLOSEUP_ITERATIONS = int(os.environ.get("BTS_CLOSEUP_ITERATIONS", "40000"))
POSITION_LR_MAX_STEPS = int(os.environ.get("BTS_POSITION_LR_MAX_STEPS", str(ITERATIONS)))
if ITERATIONS <= 0 or CLOSEUP_ITERATIONS <= 0 or POSITION_LR_MAX_STEPS <= 0:
    raise ValueError("BTS_ITERATIONS, BTS_CLOSEUP_ITERATIONS and BTS_POSITION_LR_MAX_STEPS must be positive.")
# Keep recovery/model-selection checkpoints at these milestones.  Checkpoint
# I/O is much cheaper than rendering a set of held-out cameras, and the latter
# is unnecessary once 40k has been selected as the submission schedule.
_requested_checkpoint_iterations = {
    int(value.strip())
    # With a 5M safety ceiling, a CUDA OOM is a hard crash rather than a clean
    # deadline stop.  Checkpoint at 10k/20k as well, while train.py retains
    # only the latest verified archive so this does not accumulate disk usage.
    for value in os.environ.get("BTS_CHECKPOINT_ITERATIONS", "10000,20000,30000,35000,40000,45000,50000,55000,60000").split(",")
    if value.strip()
}
CHECKPOINT_ITERATIONS = sorted(
    # Scene profiles may legitimately run beyond BTS_ITERATIONS (for example
    # chair's 70k reconstruction plus 10k cleanup).  Filter against each
    # scene's target later in scene_train_config(), not against this shared
    # default here.
    iteration for iteration in _requested_checkpoint_iterations if iteration > 0
)
if ITERATIONS not in CHECKPOINT_ITERATIONS:
    CHECKPOINT_ITERATIONS.append(ITERATIONS)
# Do not render validation views at 30k/35k.  Keep the final fixed-view WandB
# diagnostic at 40k; override only when deliberately running an ablation.
_requested_validation_iterations = {
    int(value.strip())
    for value in os.environ.get("BTS_VALIDATION_ITERATIONS", str(ITERATIONS)).split(",")
    if value.strip()
}
VALIDATION_ITERATIONS = sorted(
    iteration for iteration in _requested_validation_iterations if iteration > 0
)
if ITERATIONS not in VALIDATION_ITERATIONS:
    VALIDATION_ITERATIONS.append(ITERATIONS)
# Use full resolution for thin BTS and cable details; set it to 2 only for a constrained rerun.
TRAIN_RESOLUTION = int(os.environ.get("BTS_TRAIN_RESOLUTION", "1"))
RENDER_RESOLUTION = int(os.environ.get("BTS_RENDER_RESOLUTION", "1"))
USE_ANTIALIASING = os.environ.get("BTS_ANTIALIASING", "1").strip() != "0"
RENDER_ENSEMBLE_SCALES = [
    float(scale.strip())
    for scale in os.environ.get("BTS_RENDER_ENSEMBLE_SCALES", "1.0").split(",")
    if scale.strip()
]
if not RENDER_ENSEMBLE_SCALES or any(scale < 1.0 for scale in RENDER_ENSEMBLE_SCALES):
    raise ValueError("BTS_RENDER_ENSEMBLE_SCALES must contain one or more scales >= 1.0.")
# Close-range objects need native-resolution detail rather than the BTS
# multi-scale smoothing pass.  This remains overrideable for ablations.
CLOSEUP_RENDER_ENSEMBLE_SCALES = [
    float(scale.strip())
    for scale in os.environ.get("BTS_CLOSEUP_RENDER_ENSEMBLE_SCALES", "1.0").split(",")
    if scale.strip()
]
if not CLOSEUP_RENDER_ENSEMBLE_SCALES or any(scale < 1.0 for scale in CLOSEUP_RENDER_ENSEMBLE_SCALES):
    raise ValueError("BTS_CLOSEUP_RENDER_ENSEMBLE_SCALES must contain one or more scales >= 1.0.")
# A 7M run needs bounded densification events and lower SH storage on a T4;
# both are configured below.  It remains an upper bound, not a point target.
MAX_GAUSSIANS = int(os.environ.get("BTS_MAX_GAUSSIANS", "8000000"))
SH_DEGREE = int(os.environ.get("BTS_SH_DEGREE", "2"))
MAX_NEW_POINTS_PER_DENSIFY = int(os.environ.get("BTS_MAX_NEW_POINTS_PER_DENSIFY", "75000"))
# Thin tower lattice and cables retain high image-space gradients late in
# training.  Stopping at 15k freezes their allocation while the 15k--40k
# phase merely optimises oversized splats, producing the observed smearing.
DENSIFY_GRAD_THRESHOLD = float(os.environ.get("BTS_DENSIFY_GRAD_THRESHOLD", "0.00015"))
DENSIFY_UNTIL_ITER = int(os.environ.get("BTS_DENSIFY_UNTIL_ITER", "50000"))
PERCENT_DENSE = float(os.environ.get("BTS_PERCENT_DENSE", "0.005"))
if DENSIFY_GRAD_THRESHOLD <= 0 or DENSIFY_UNTIL_ITER <= 0 or not 0 < PERCENT_DENSE <= 1:
    raise ValueError("BTS densification settings must be positive, and BTS_PERCENT_DENSE must be in (0, 1].")
FOREGROUND_LOSS_WEIGHT = float(os.environ.get("BTS_FOREGROUND_LOSS_WEIGHT", "12.0"))
FOREGROUND_EDGE_LOSS_WEIGHT = float(os.environ.get("BTS_FOREGROUND_EDGE_LOSS_WEIGHT", "0.05"))
DISABLE_FOREGROUND_MASK_SCENES = frozenset(
    name.strip()
    for name in os.environ.get("BTS_DISABLE_FOREGROUND_MASK_SCENES", "").split(",")
    if name.strip()
)
# A quality profile may demand that masks exist for a useful subset of views,
# rather than silently falling back to full-image loss if a Kaggle input was
# mounted without the mask directory.
REQUIRE_FOREGROUND_MASK_SCENES = frozenset(
    name.strip()
    for name in os.environ.get("BTS_REQUIRE_FOREGROUND_MASK_SCENES", "").split(",")
    if name.strip()
)
MIN_FOREGROUND_MASK_COVERAGE = float(os.environ.get("BTS_MIN_FOREGROUND_MASK_COVERAGE", "0.0"))
MIN_DEPTH_COVERAGE = float(os.environ.get("BTS_MIN_DEPTH_COVERAGE", "0.10"))
MAX_WORKERS = int(os.environ.get("BTS_MAX_WORKERS", "2"))
KAGGLE_TIME_LIMIT_H = float(os.environ.get("BTS_TIME_LIMIT_H", "11.5"))
# Reserve time for the final compact checkpoint, render, and packaging.
KAGGLE_STOP_BUFFER_MIN = float(os.environ.get("BTS_STOP_BUFFER_MIN", "30"))
# Below this threshold, training exits cleanly and preserves the last verified checkpoint.
MIN_FREE_DISK_GB = float(os.environ.get("BTS_MIN_FREE_DISK_GB", "2.0"))
DISK_CHECK_INTERVAL = int(os.environ.get("BTS_DISK_CHECK_INTERVAL", "100"))
WANDB_LOG_INTERVAL = int(os.environ.get("BTS_WANDB_LOG_INTERVAL", "100"))
SUBPROCESS_HEARTBEAT_SECONDS = float(os.environ.get("BTS_SUBPROCESS_HEARTBEAT_SECONDS", "30"))
# Fixed train-view monitor used for model selection.  Test poses have no ground
# truth, so this is the only image-space signal available during a Kaggle run.
# By default the monitor views remain in training: novel-view coverage matters
# more than a strict holdout for the final competition reconstruction.
VALIDATION_FRACTION = float(os.environ.get("BTS_VALIDATION_FRACTION", "0.05"))
VALIDATION_HOLDOUT = os.environ.get("BTS_VALIDATION_HOLDOUT", "0").strip() == "1"
VALIDATION_LPIPS_FINAL = os.environ.get("BTS_VALIDATION_LPIPS_FINAL", "0").strip() == "1"
# Build a writable, camera-consistent copy only for explicitly selected BTS
# scenes.  The original Kaggle input remains untouched, and the default keeps
# legacy behaviour for existing runs/checkpoints.
PINHOLE_PREPROCESS_SCENES = frozenset(
    name.strip()
    for name in os.environ.get("BTS_PINHOLE_PREPROCESS_SCENES", "").split(",")
    if name.strip()
)
PINHOLE_DATA_ROOT = Path(os.environ.get("BTS_PINHOLE_DATA_ROOT", "/kaggle/working/data_pinhole"))
PINHOLE_JPEG_QUALITY = int(os.environ.get("BTS_PINHOLE_JPEG_QUALITY", "100"))
# Resume an interrupted run from its own output by default.  Input checkpoints
# can be from a different data/preprocessing/configuration generation, so they
# are deliberately opt-in for quality-sensitive reruns.
RESUME_LOCAL = os.environ.get("BTS_RESUME_LOCAL", "1").strip() != "0"
RESUME_INPUT = os.environ.get("BTS_RESUME_INPUT", "0").strip() == "1"
# An explicit fresh run clears only the selected scene output/submission
# directories immediately before that scene starts.  It is opt-in because it
# intentionally discards resumable checkpoints from a prior experiment.
FRESH_RUN = os.environ.get("BTS_FRESH_RUN", "0").strip() == "1"
# A named run creates a new one-reset marker: changed profiles can start
# clean, while an interrupted run of that same profile remains resumable.
FRESH_RUN_ID = os.environ.get("BTS_FRESH_RUN_ID", "").strip()
# Chair has failed geometry in the current submission, so it is the only
# scene retrained from scratch by the quality profile.  A durable fresh-run
# marker makes an interrupted 70k run resume rather than deleting itself.
FRESH_SCENES = frozenset(
    name.strip() for name in os.environ.get("BTS_FRESH_SCENES", "chair").split(",") if name.strip()
)
# Existing 70k models should not be sent through another generic training
# tail.  They receive a short, geometry-conservative alignment phase instead:
# no new splats, fresh visibility evidence, and prune-only cleanup.  A
# fine-tune is unsafe without an input/local checkpoint, so build_train_cmd()
# rejects an accidental fresh start for these scenes.
FINETUNE_SCENES = frozenset(
    name.strip()
    for name in os.environ.get(
        "BTS_FINETUNE_SCENES", "bonsai,HCM0421,HCM0539,HCM0540,HCM0644,HCM0674"
    ).split(",")
    if name.strip()
)
# Detail-recovery runs need a checkpoint but must not inherit FINETUNE_SCENES,
# whose policy intentionally turns densification off and enables prune-only
# cleanup.  Keep the safety guard independent from that old phase policy.
REQUIRE_RESUME_SCENES = frozenset(
    name.strip() for name in os.environ.get("BTS_REQUIRE_RESUME_SCENES", "").split(",") if name.strip()
)
REQUIRE_RESUME_MIN_ITERATION = int(os.environ.get("BTS_REQUIRE_RESUME_MIN_ITERATION", "0"))
if REQUIRE_RESUME_MIN_ITERATION < 0:
    raise ValueError("BTS_REQUIRE_RESUME_MIN_ITERATION must be non-negative.")
# Archive-only restores are a stricter contract than a generic resume.  They
# are used by a full submission profile to guarantee that a close-up scene is
# rendered from its selected checkpoint rather than an accidental raw PTH.
REQUIRE_CHECKPOINT_ARCHIVE_SCENES = frozenset(
    name.strip()
    for name in os.environ.get("BTS_REQUIRE_CHECKPOINT_ARCHIVE_SCENES", "").split(",")
    if name.strip()
)
FINETUNE_BASE_ITERATIONS = int(os.environ.get("BTS_FINETUNE_BASE_ITERATIONS", "70000"))
FINETUNE_STEPS = int(os.environ.get("BTS_FINETUNE_STEPS", "10000"))
# train.py's position schedule uses absolute iteration numbers.  Extending
# its horizon prevents a 70k checkpoint from receiving a near-zero geometry
# learning rate during the alignment tail.
FINETUNE_POSITION_LR_MAX_STEPS = int(
    os.environ.get("BTS_FINETUNE_POSITION_LR_MAX_STEPS", "140000")
)
if FINETUNE_BASE_ITERATIONS <= 0 or FINETUNE_STEPS <= 0 or FINETUNE_POSITION_LR_MAX_STEPS <= 0:
    raise ValueError("BTS_FINETUNE_* iteration and LR-horizon values must be positive.")
# Chair receives a full fresh schedule even if BTS_CLOSEUP_ITERATIONS is set
# lower for a separate bonsai ablation.
CHAIR_FRESH_ITERATIONS = int(os.environ.get("BTS_CHAIR_FRESH_ITERATIONS", "70000"))
if CHAIR_FRESH_ITERATIONS <= 0:
    raise ValueError("BTS_CHAIR_FRESH_ITERATIONS must be positive.")
# Comma-separated scenes that should resume a completed model into a
# prune-only cleanup phase.  This is intentionally opt-in so normal training
# runs do not extend their target iteration.
CLEANUP_SCENES = frozenset(
    name.strip() for name in os.environ.get("BTS_CLEANUP_SCENES", "").split(",") if name.strip()
)
# A render-only scene must already have a checkpoint/PLY at its configured
# target iteration.  This makes a final rendering pass safe: a missing or
# incorrectly-versioned archive can never silently start a costly retrain.
RENDER_ONLY_SCENES = frozenset(
    name.strip() for name in os.environ.get("BTS_RENDER_ONLY_SCENES", "").split(",") if name.strip()
)
# Run these scenes to completion before the remaining queue is allowed to
# hydrate checkpoints and render.  It is opt-in: the normal mixed queue keeps
# its historical priority order unless a run profile explicitly requests it.
TRAIN_FIRST_SCENES = frozenset(
    name.strip() for name in os.environ.get("BTS_TRAIN_FIRST_SCENES", "").split(",") if name.strip()
)
# In exclusive mode, later scenes wait until every train-first scene ends.
# Disable it for 2x-GPU runs to launch the priority scene first while keeping
# the other T4 busy with the next queued scene.
TRAIN_FIRST_EXCLUSIVE = os.environ.get("BTS_TRAIN_FIRST_EXCLUSIVE", "1").strip() != "0"
BTS_CLEANUP_STEPS = int(os.environ.get("BTS_CLEANUP_STEPS", "0"))
CLOSEUP_CLEANUP_STEPS = int(os.environ.get("BTS_CLOSEUP_CLEANUP_STEPS", "0"))
if BTS_CLEANUP_STEPS < 0 or CLOSEUP_CLEANUP_STEPS < 0:
    raise ValueError("BTS_CLEANUP_STEPS and BTS_CLOSEUP_CLEANUP_STEPS must be non-negative.")
# Successful scenes normally release their model after their exact render set
# was copied, saving Kaggle disk.  Enable this for a rerun when final PLY and
# checkpoint artifacts must remain available for inspection or later renders.
KEEP_MODEL_ARTIFACTS = os.environ.get("BTS_KEEP_MODEL_ARTIFACTS", "0").strip() == "1"
# When non-empty, retain final model artifacts only for these scenes.  This is
# useful for inspecting a newly retrained close-up pair without keeping five
# large hydrated BTS models after their render-only phase.
KEEP_MODEL_SCENES = frozenset(
    name.strip() for name in os.environ.get("BTS_KEEP_MODEL_SCENES", "").split(",") if name.strip()
)
# Offset checkpoint writes per GPU so two large archives do not saturate the
# small Kaggle working disk at the same iteration.
CHECKPOINT_STAGGER_SECONDS = float(os.environ.get("BTS_CHECKPOINT_STAGGER_SECONDS", "90"))
# Retaining every 5M-Gaussian checkpoint can fill Kaggle's working disk.  Keep
# the two newest verified backups per scene (for example 35000 and 40000).
CHECKPOINT_BACKUP_KEEP = int(os.environ.get("BTS_CHECKPOINT_BACKUP_KEEP", "2"))
# Store a portable zip of the extracted checkpoint archive after every clean
# train exit, including deadline/disk early-stop.  Keeping only the zip avoids
# retaining two full copies of a 10M-Gaussian checkpoint on Kaggle disk.
CHECKPOINT_ARCHIVE_ZIP = os.environ.get("BTS_CHECKPOINT_ARCHIVE_ZIP", "1").strip() != "0"
if (
    TRAIN_RESOLUTION <= 0
    or RENDER_RESOLUTION <= 0
    or MAX_GAUSSIANS < 0
    or SH_DEGREE < 0
    or SH_DEGREE > 3
    or MAX_NEW_POINTS_PER_DENSIFY < 0
    or FOREGROUND_LOSS_WEIGHT < 0
    or FOREGROUND_EDGE_LOSS_WEIGHT < 0
    or not 0.0 <= MIN_DEPTH_COVERAGE <= 1.0
    or MAX_WORKERS <= 0
    or KAGGLE_TIME_LIMIT_H <= 0
    or KAGGLE_STOP_BUFFER_MIN < 0
    or MIN_FREE_DISK_GB < 0
    or DISK_CHECK_INTERVAL <= 0
    or WANDB_LOG_INTERVAL <= 0
    or SUBPROCESS_HEARTBEAT_SECONDS <= 0
    or not 0.0 < VALIDATION_FRACTION < 1.0
    or CHECKPOINT_STAGGER_SECONDS < 0
    or CHECKPOINT_BACKUP_KEEP <= 0
):
    raise ValueError(
        "Invalid BTS configuration: resolutions/workers/intervals must be positive; "
        "weights, disk threshold, and checkpoint stagger must be non-negative; "
        "heartbeat interval must be positive; "
        "BTS_VALIDATION_FRACTION must be in (0, 1)."
    )
TARGET_SCENES = [
    "bonsai",
    "chair",
    "HCM0421",
    "HCM0539",
    "HCM0540",
    "HCM0644",
    "HCM0674",
]
CLOSEUP_SCENES = frozenset({"bonsai", "chair"})
SCENE_FILTER = os.environ.get("BTS_SCENES", ",".join(TARGET_SCENES)).strip()
_unknown_control_scenes = (RENDER_ONLY_SCENES | TRAIN_FIRST_SCENES) - set(TARGET_SCENES)
if _unknown_control_scenes:
    raise ValueError(f"Unknown BTS control scenes: {sorted(_unknown_control_scenes)}")
_unknown_mask_scenes = DISABLE_FOREGROUND_MASK_SCENES - set(TARGET_SCENES)
if _unknown_mask_scenes:
    raise ValueError(f"Unknown BTS_DISABLE_FOREGROUND_MASK_SCENES: {sorted(_unknown_mask_scenes)}")
_unknown_required_mask_scenes = REQUIRE_FOREGROUND_MASK_SCENES - set(TARGET_SCENES)
if _unknown_required_mask_scenes:
    raise ValueError(
        f"Unknown BTS_REQUIRE_FOREGROUND_MASK_SCENES: {sorted(_unknown_required_mask_scenes)}"
    )
if not 0.0 <= MIN_FOREGROUND_MASK_COVERAGE <= 1.0:
    raise ValueError("BTS_MIN_FOREGROUND_MASK_COVERAGE must be in [0, 1].")
if RENDER_ONLY_SCENES & FRESH_SCENES:
    raise ValueError(
        "A scene cannot be both BTS_RENDER_ONLY_SCENES and BTS_FRESH_SCENES: "
        f"{sorted(RENDER_ONLY_SCENES & FRESH_SCENES)}"
    )
_unknown_finetune_scenes = FINETUNE_SCENES - set(TARGET_SCENES)
if _unknown_finetune_scenes:
    raise ValueError(f"Unknown BTS_FINETUNE_SCENES: {sorted(_unknown_finetune_scenes)}")
_unknown_required_resume_scenes = REQUIRE_RESUME_SCENES - set(TARGET_SCENES)
if _unknown_required_resume_scenes:
    raise ValueError(
        f"Unknown BTS_REQUIRE_RESUME_SCENES: {sorted(_unknown_required_resume_scenes)}"
    )
_unknown_required_archive_scenes = REQUIRE_CHECKPOINT_ARCHIVE_SCENES - set(TARGET_SCENES)
if _unknown_required_archive_scenes:
    raise ValueError(
        f"Unknown BTS_REQUIRE_CHECKPOINT_ARCHIVE_SCENES: {sorted(_unknown_required_archive_scenes)}"
    )
if FINETUNE_SCENES & FRESH_SCENES:
    raise ValueError(
        "A scene cannot be both BTS_FINETUNE_SCENES and BTS_FRESH_SCENES: "
        f"{sorted(FINETUNE_SCENES & FRESH_SCENES)}"
    )
_unknown_keep_scenes = KEEP_MODEL_SCENES - set(TARGET_SCENES)
if _unknown_keep_scenes:
    raise ValueError(f"Unknown BTS_KEEP_MODEL_SCENES: {sorted(_unknown_keep_scenes)}")
_unknown_pinhole_scenes = PINHOLE_PREPROCESS_SCENES - set(TARGET_SCENES)
if _unknown_pinhole_scenes:
    raise ValueError(f"Unknown BTS_PINHOLE_PREPROCESS_SCENES: {sorted(_unknown_pinhole_scenes)}")
if not 1 <= PINHOLE_JPEG_QUALITY <= 100:
    raise ValueError("BTS_PINHOLE_JPEG_QUALITY must be in [1, 100].")

def get_secret(name):
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name).strip()
    except Exception:
        return ""


def get_first_secret(names):
    for name in names:
        value = get_secret(name)
        if value:
            print(f"Loaded secret: {name}")
            return value
    return ""

#public wandb key. DO NOT CHANGE IT.
WANDB_API_KEY = "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"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "ai_race").strip()
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "bts-digital-twin")
WANDB_REQUIRED = os.environ.get("BTS_REQUIRE_WANDB", "1").strip() != "0"
USE_WANDB = bool(WANDB_API_KEY)
if USE_WANDB:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    os.environ.setdefault("WANDB_MODE", "online")
    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

SESSION_START = time.time()
TRAIN_STOP_AT_UNIX_TIME = SESSION_START + max(
    0.0, KAGGLE_TIME_LIMIT_H * 3600 - KAGGLE_STOP_BUFFER_MIN * 60,
)

# The main notebook thread may receive KeyboardInterrupt while workers are
# blocking in a child train.py process.  Keep those PIDs so the interrupt path
# can stop them before ThreadPoolExecutor waits for its worker threads.
_ACTIVE_PROCESSES = set()
_ACTIVE_PROCESSES_LOCK = threading.Lock()


def stop_active_processes():
    """Terminate active train/render children so an interrupted cell can exit."""
    with _ACTIVE_PROCESSES_LOCK:
        processes = list(_ACTIVE_PROCESSES)
    for process in processes:
        if process.poll() is None:
            print(f"Stopping subprocess pid={process.pid} after notebook interrupt.")
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            print(f"Killing unresponsive subprocess pid={process.pid}.")
            process.kill()


def run(cmd, cwd=None, log_file=None, check=False, env=None, stream=False):
    """Run a command, optionally teeing a persisted log into the notebook."""
    cwd = str(cwd) if cwd is not None else None
    printable = " ".join(str(x) for x in cmd)
    print(f"\n$ {printable}")
    merged_env = os.environ.copy()
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    if env:
        merged_env.update(env)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8", errors="replace") as f:
            if stream:
                process = subprocess.Popen(
                    [str(x) for x in cmd],
                    cwd=cwd,
                    env=merged_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                )
                assert process.stdout is not None
                with _ACTIVE_PROCESSES_LOCK:
                    _ACTIVE_PROCESSES.add(process)

                # A blocking os.read() gave the notebook no indication of a
                # child stuck before its first print, and made Interrupt leave
                # the executor waiting for its worker.  Drain stdout in a
                # reader thread so the worker can emit a useful heartbeat.
                output_queue = queue.Queue()

                def drain_stdout():
                    try:
                        while True:
                            chunk = process.stdout.read(4096)
                            if not chunk:
                                break
                            output_queue.put(chunk)
                    finally:
                        output_queue.put(None)

                reader = threading.Thread(target=drain_stdout, name=f"stream-{process.pid}", daemon=True)
                reader.start()
                started = time.monotonic()
                try:
                    while True:
                        try:
                            chunk = output_queue.get(timeout=SUBPROCESS_HEARTBEAT_SECONDS)
                        except queue.Empty:
                            if process.poll() is None:
                                elapsed = time.monotonic() - started
                                print(
                                    f"[pid={process.pid}] still running for {elapsed / 60:.1f} min; "
                                    f"waiting for subprocess output. Log: {log_file}",
                                    flush=True,
                                )
                            continue
                        if chunk is None:
                            break
                        output = chunk.decode("utf-8", errors="replace")
                        f.write(output)
                        f.flush()
                        print(output, end="", flush=True)
                    result = process.wait()
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                    raise
                finally:
                    with _ACTIVE_PROCESSES_LOCK:
                        _ACTIVE_PROCESSES.discard(process)
                    process.stdout.close()
                    reader.join(timeout=1)
            else:
                result = subprocess.run(
                    [str(x) for x in cmd],
                    cwd=cwd,
                    env=merged_env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                ).returncode
    else:
        result = subprocess.run(
            [str(x) for x in cmd],
            cwd=cwd,
            env=merged_env,
            text=True,
            capture_output=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

    returncode = result if log_file else result.returncode
    if check and returncode != 0:
        raise RuntimeError(f"Command failed with rc={returncode}: {printable}")
    return returncode


def tail(path, n=80):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception as exc:
        return f"<could not read {path}: {exc}>"


def disk_free_gb(path="/kaggle/working"):
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3), total / (1024**3)


def hours_remaining():
    elapsed = (time.time() - SESSION_START) / 3600.0
    return KAGGLE_TIME_LIMIT_H - elapsed


print("=" * 80)
print("Environment")
print("=" * 80)
run(["nvidia-smi"])
run([sys.executable, "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"])
free_gb, total_gb = disk_free_gb()
print(f"Disk: {free_gb:.1f} GB free / {total_gb:.1f} GB total")


# =============================================================================
# CELL 2 - Repo and dependencies
# =============================================================================

REPO_SYNC_REQUIRED = os.environ.get("BTS_REQUIRE_REPO_SYNC", "0").strip() == "1"

if REPO_DIR.exists():
    # Never continue with a stale checkout: it can silently omit new training
    # flags even though this notebook file itself has been updated.  A prior
    # notebook run can leave harmless compatibility edits in this checkout,
    # however; a failed fast-forward must not prevent the training queue from
    # starting.  Set BTS_REQUIRE_REPO_SYNC=1 when using an exact revision is
    # more important than using the available checkout.
    sync_rc = run(["git", "-C", REPO_DIR, "fetch", "origin", REPO_REF], check=False)
    if sync_rc == 0:
        sync_rc = run(["git", "-C", REPO_DIR, "merge", "--ff-only", f"origin/{REPO_REF}"], check=False)
    if sync_rc != 0:
        message = (
            f"Repository sync failed (rc={sync_rc}); continuing with existing checkout "
            f"at {REPO_DIR}. Set BTS_REQUIRE_REPO_SYNC=1 to fail instead."
        )
        if REPO_SYNC_REQUIRED:
            raise RuntimeError(message)
        print(f"WARNING: {message}")
    submodule_rc = run(["git", "-C", REPO_DIR, "submodule", "update", "--init", "--recursive"], check=False)
    if submodule_rc != 0:
        message = f"Submodule update failed (rc={submodule_rc})."
        if REPO_SYNC_REQUIRED:
            raise RuntimeError(message)
        print(f"WARNING: {message} Continuing with the existing submodules.")
else:
    run(["git", "clone", "--branch", REPO_REF, "--recurse-submodules", REPO_URL, REPO_DIR], check=True)

os.chdir(REPO_DIR)
run(["git", "log", "--oneline", "-3"], cwd=REPO_DIR, check=False)
run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, check=True)


def patch_repo_for_kaggle():
    """Apply tiny compatibility patches before train.py builds argparse."""
    arguments_file = REPO_DIR / "arguments" / "__init__.py"
    text = arguments_file.read_text(encoding="utf-8", errors="replace")
    patched = text.replace("self._masks = \"\"", "self.masks = \"\"")
    if patched != text:
        arguments_file.write_text(patched, encoding="utf-8")
        print("Patched arguments/__init__.py: masks no longer reserves shorthand -m.")

    cameras_file = REPO_DIR / "scene" / "cameras.py"
    text = cameras_file.read_text(encoding="utf-8", errors="replace")
    patched = text.replace("                    self.depth_mask *= 0\n", "")
    if patched != text:
        cameras_file.write_text(patched, encoding="utf-8")
        print("Patched scene/cameras.py: depth_mask is no longer touched before initialization.")


patch_repo_for_kaggle()

print("=" * 80)
print("Installing Python deps and CUDA extensions")
print("=" * 80)

run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
# setuptools<70 must be pinned BEFORE CUDA extensions are built to avoid
# distutils removal breakage (setuptools>=70 drops distutils shim).
# wheel and packaging are required build-time deps for --no-build-isolation.
run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
     "setuptools<70.0.0", "wheel", "packaging", "ninja",
     "plyfile", "tqdm", "opencv-python-headless", "Pillow", "matplotlib"], check=True)
if USE_WANDB:
    run([sys.executable, "-m", "pip", "install", "-q", "wandb"], check=True)

# Print build environment info for diagnostics
run([sys.executable, "-c",
     "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda); "
     "import setuptools; print('setuptools', setuptools.__version__)"], check=False)
run(["nvcc", "--version"], check=False)

def _install_cuda_ext(submodule):
    """Install a CUDA extension reliably without pip network hangs.

    Strategy:
      1. Compile with 'setup.py build_ext --inplace' — pure local build,
         no network calls, no pip locks. This avoids the Kaggle pip hang
         that occurs when resolving dependencies for the 2nd+ extension.
      2. Register with 'pip install --no-build-isolation --no-cache-dir
         --no-deps .' so the package lands in site-packages and is importable
         in subprocesses (train.py, render.py) without PYTHONPATH tricks.
      3. If inplace build fails, try a full pip install as last resort.
    """
    log_file = OUTPUT_DIR / f"install_{submodule.name}.log"
    arch_list = "7.5;8.0;8.6;8.9;9.0"
    build_env = {"MAX_JOBS": "4", "TORCH_CUDA_ARCH_LIST": arch_list}

    # Step 1: compile the CUDA .so in-place (no network, no pip locks)
    rc = run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=submodule, log_file=log_file, check=False, env=build_env,
    )
    if rc != 0:
        # Retry single-threaded in case of parallel compilation race
        print(f"[{submodule.name}] build_ext failed (rc={rc}), retrying MAX_JOBS=1 ...")
        rc = run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=submodule, log_file=log_file, check=False,
            env={"MAX_JOBS": "1", "TORCH_CUDA_ARCH_LIST": arch_list},
        )

    if rc == 0:
        # Step 2: register in site-packages using --no-deps (no network calls)
        rc2 = run(
            [sys.executable, "-m", "pip", "install",
             "--no-build-isolation", "--no-cache-dir", "--no-deps", str(submodule)],
            cwd=REPO_DIR, log_file=log_file, check=False,
            env={"MAX_JOBS": "1", "TORCH_CUDA_ARCH_LIST": arch_list},
        )
        if rc2 == 0:
            print(f"[{submodule.name}] installed OK (build_ext --inplace + pip --no-deps)")
            return
        # pip registration failed but .so is built — add source dir to path as fallback
        print(f"[{submodule.name}] pip --no-deps failed (rc={rc2}), adding to sys.path directly")
        src = str(submodule)
        if src not in sys.path:
            sys.path.insert(0, src)
        # Persist path for subprocesses via .pth file
        import site
        pth = Path(site.getsitepackages()[0]) / f"_bts_{submodule.name}.pth"
        pth.write_text(src + "\n", encoding="utf-8")
        print(f"[{submodule.name}] registered via {pth}")
        return

    # All methods failed
    log_tail = tail(log_file, 80)
    print(f"ERROR: failed to build {submodule.name}.\n--- build log ---\n{log_tail}\n--- end ---")
    raise RuntimeError(f"Could not build {submodule.name}. Full log: {log_file}")

_submodules = [
    REPO_DIR / "submodules" / "diff-gaussian-rasterization",
    REPO_DIR / "submodules" / "simple-knn",
    REPO_DIR / "submodules" / "fused-ssim",
]
for submodule in _tqdm(_submodules, desc="CUDA extensions", unit="ext"):
    if submodule.exists():
        _install_cuda_ext(submodule)
    else:
        print(f"WARNING: missing submodule {submodule}")

verify_code = (
    "from diff_gaussian_rasterization import GaussianRasterizer; "
    "from simple_knn._C import distCUDA2; "
    "from fused_ssim import fused_ssim; "
    "print('CUDA extensions OK: rasterizer, simple-knn, fused-ssim')"
)
run([sys.executable, "-c", verify_code], cwd=REPO_DIR, check=True)

if USE_WANDB:
    try:
        import wandb

        wandb.login(key=WANDB_API_KEY)
        if not WANDB_ENTITY:
            try:
                WANDB_ENTITY = wandb.Api().viewer.username
            except Exception:
                WANDB_ENTITY = ""
        print(f"WandB enabled: project={WANDB_PROJECT}, entity={WANDB_ENTITY or '<default>'}")
    except Exception as exc:
        message = f"WandB login failed: {exc}"
        USE_WANDB = False
        if WANDB_REQUIRED:
            raise RuntimeError(message) from exc
        print(f"WARNING: {message}. Continuing with WandB disabled.")
else:
    message = "WandB API key not found. Add Kaggle Secret named WANDB_API_KEY to enable logging."
    if WANDB_REQUIRED:
        raise RuntimeError(message)
    print(f"WandB disabled. {message}")

ARGUMENTS_TEXT = (REPO_DIR / "arguments" / "__init__.py").read_text(encoding="utf-8", errors="replace")
SUPPORTS_MAX_GAUSSIANS = "max_gaussians" in ARGUMENTS_TEXT
SUPPORTS_MASKS = "self.masks" in ARGUMENTS_TEXT or "_masks" in ARGUMENTS_TEXT or "self._masks" in ARGUMENTS_TEXT
SUPPORTS_FOREGROUND_WEIGHT = "foreground_loss_weight" in ARGUMENTS_TEXT
print(
    "Repo feature flags:",
    {
        "max_gaussians": SUPPORTS_MAX_GAUSSIANS,
        "masks": SUPPORTS_MASKS,
        "foreground_loss_weight": SUPPORTS_FOREGROUND_WEIGHT,
    },
)


# =============================================================================
# CELL 3 - Data discovery and scene diagnostics
# =============================================================================

def find_data_root():
    for candidate in DATA_ROOT_CANDIDATES:
        if candidate and candidate.exists():
            # Support both the legacy phase1/private_set1 mount and the new
            # flat dataset mount.  Validate target names here so a generic
            # Kaggle input directory is never selected by accident.
            scene_base = candidate / "private_set1" if (candidate / "private_set1").exists() else candidate
            if any((scene_base / name).is_dir() for name in TARGET_SCENES):
                return candidate
    raise FileNotFoundError(
        "Could not find the dataset root. Set BTS_DATA_DIR to the folder containing the seven scene directories."
    )


def is_scene_dir(path):
    path = Path(path)
    return (path / "train" / "sparse").exists() or (path / "sparse").exists()


def discover_scenes(data_root):
    split_dir = data_root / "private_set1" if (data_root / "private_set1").exists() else data_root

    if SCENE_FILTER:
        wanted = {x.strip() for x in SCENE_FILTER.split(",") if x.strip()}
    else:
        wanted = set(TARGET_SCENES)

    unknown = sorted(wanted - set(TARGET_SCENES))
    if unknown:
        raise ValueError(f"BTS_SCENES contains names outside the target scene list: {unknown}")

    scenes = [split_dir / name for name in TARGET_SCENES if name in wanted]
    missing_dirs = [p.name for p in scenes if not p.exists()]
    invalid_dirs = [p.name for p in scenes if p.exists() and not is_scene_dir(p)]
    if missing_dirs:
        raise FileNotFoundError(f"Missing target scene dirs: {missing_dirs}")
    if invalid_dirs:
        raise RuntimeError(f"Invalid private_set1 scene dirs, missing train/sparse or sparse: {invalid_dirs}")

    if not scenes:
        raise RuntimeError(f"No selected target scenes found under {split_dir}")
    return scenes


DATA_ROOT = find_data_root()
ALL_SCENES = discover_scenes(DATA_ROOT)
print(f"DATA_ROOT: {DATA_ROOT}")
print(f"Target scenes ({len(ALL_SCENES)}): {[p.name for p in ALL_SCENES]}")


def prepare_selected_pinhole_scenes(scenes):
    """Return scene paths, replacing opted-in radial scenes with pinhole copies."""
    if not PINHOLE_PREPROCESS_SCENES:
        return scenes
    script = REPO_DIR / "prepare_pinhole_dataset.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing pinhole preprocessing script: {script}")
    prepared = []
    for scene in scenes:
        if scene.name not in PINHOLE_PREPROCESS_SCENES:
            prepared.append(scene)
            continue
        destination = PINHOLE_DATA_ROOT / scene.name
        print(f"[{scene.name}] preparing radial->pinhole training copy at {destination}")
        run(
            [sys.executable, script, "--source", scene, "--destination", destination,
             "--jpeg-quality", str(PINHOLE_JPEG_QUALITY)],
            cwd=REPO_DIR,
            log_file=OUTPUT_DIR / f"{scene.name}_pinhole_preprocess.log",
            check=True,
        )
        manifest = destination / ".pinhole_manifest.json"
        if not manifest.is_file() or not (destination / "train" / "sparse" / "0" / "cameras.bin").is_file():
            raise RuntimeError(f"[{scene.name}] pinhole preprocessing did not create a valid scene at {destination}")
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"[{scene.name}] could not read pinhole manifest {manifest}: {exc}") from exc
        if metadata.get("camera_model") != "PINHOLE":
            raise RuntimeError(f"[{scene.name}] pinhole manifest does not declare PINHOLE camera model.")
        print(
            f"[{scene.name}] PINHOLE ACTIVE: source={destination} | "
            f"images={metadata.get('image_count')} | masks={metadata.get('mask_counts', {})}"
        )
        prepared.append(destination)
    return prepared


ALL_SCENES = prepare_selected_pinhole_scenes(ALL_SCENES)
if PINHOLE_PREPROCESS_SCENES:
    print(f"Pinhole-preprocessed scenes: {sorted(PINHOLE_PREPROCESS_SCENES)}")


def train_root(scene_path):
    scene_path = Path(scene_path)
    return scene_path / "train" if (scene_path / "train" / "sparse").exists() else scene_path


def count_images(scene_path):
    image_dir = train_root(scene_path) / "images"
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    if not image_dir.exists():
        return 0
    return sum(1 for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def optional_depth_args(scene_path):
    root = train_root(scene_path)
    depth_params = root / "sparse" / "0" / "depth_params.json"
    if not depth_params.exists():
        print(f"[{Path(scene_path).name}] no Depth Anything metadata; depth regularization is disabled.")
        return []
    try:
        params = json.loads(depth_params.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[{Path(scene_path).name}] invalid depth_params.json ({exc}); depth regularization is disabled.")
        return []
    if not isinstance(params, dict):
        print(f"[{Path(scene_path).name}] depth_params.json is not an object; depth regularization is disabled.")
        return []
    image_stems = {path.stem for path in (root / "images").glob("*") if path.is_file()}
    # A quality-gated map directory is preferred whenever it exists.  It
    # deliberately contains only views whose monocular depth agreed with
    # sparse COLMAP, so unreliable views receive no depth loss at all.
    for name in ["depths_any_reliable", "depths_any", "depth_anything", "depths", "depth"]:
        depth_dir = root / name
        if not depth_dir.is_dir():
            continue
        matched = []
        for path in depth_dir.glob("*.png"):
            entry = params.get(path.stem)
            if path.stem not in image_stems or not isinstance(entry, dict):
                continue
            try:
                scale = float(entry["scale"])
                offset = float(entry["offset"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(scale) and math.isfinite(offset) and scale > 0.0:
                matched.append(path.stem)
        coverage = len(matched) / max(1, len(image_stems))
        if coverage < MIN_DEPTH_COVERAGE:
            print(
                f"[{Path(scene_path).name}] depth prior {name}: {len(matched)}/{len(image_stems)} "
                f"({coverage:.1%}) below minimum {MIN_DEPTH_COVERAGE:.1%}; disabled."
            )
            continue
        print(
            f"[{Path(scene_path).name}] using quality-gated depth prior {name}: "
            f"{len(matched)}/{len(image_stems)} views ({coverage:.1%})."
        )
        return ["--depths", name]
    print(f"[{Path(scene_path).name}] Depth Anything metadata exists but no depth map directory was found; depth regularization is disabled.")
    return []


def optional_mask_args(scene_path):
    scene_name = Path(scene_path).name
    scene_prefix = f"BTS_{scene_name.upper()}"
    foreground_loss_weight = float(os.environ.get(
        f"{scene_prefix}_FOREGROUND_LOSS_WEIGHT", str(FOREGROUND_LOSS_WEIGHT)
    ))
    foreground_edge_loss_weight = float(os.environ.get(
        f"{scene_prefix}_FOREGROUND_EDGE_LOSS_WEIGHT", str(FOREGROUND_EDGE_LOSS_WEIGHT)
    ))
    if foreground_loss_weight < 0.0 or foreground_edge_loss_weight < 0.0:
        raise ValueError(f"[{scene_name}] foreground loss weights must be non-negative.")
    if scene_name in DISABLE_FOREGROUND_MASK_SCENES:
        print(f"[{scene_name}] foreground masks explicitly disabled for this scene.")
        return []
    if not SUPPORTS_MASKS:
        return []
    root = train_root(scene_path)
    # Prefer the generated object masks over a generic legacy ``masks``
    # directory when both are present in a Kaggle dataset mount.
    for name in ["foreground_masks", "masks", "mask", "foreground"]:
        mask_root = root / name
        if mask_root.is_dir():
            image_stems = {
                path.stem for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
                for path in (root / "images").glob(ext)
            }
            mask_stems = {
                path.stem for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
                for path in mask_root.glob(ext)
            }
            missing = image_stems - mask_stems
            if not image_stems:
                print(
                    f"[{Path(scene_path).name}] foreground masks disabled: "
                    "no training images were found."
                )
                return []
            if missing:
                print(
                    f"[{Path(scene_path).name}] foreground masks: {len(mask_stems)}/{len(image_stems)} "
                    f"matched; {len(missing)} views have no mask and will receive zero mask loss."
                )
            args = ["--masks", name]
            if SUPPORTS_FOREGROUND_WEIGHT:
                args.extend(["--foreground_loss_weight", str(foreground_loss_weight)])
                args.extend(["--foreground_edge_loss_weight", str(foreground_edge_loss_weight)])
            print(
                f"[{Path(scene_path).name}] foreground masks: {root / name} "
                f"(weight={foreground_loss_weight}, edge_weight={foreground_edge_loss_weight})"
            )
            return args
    print(f"[{Path(scene_path).name}] no foreground masks found; using image/depth losses only.")
    return []


for scene in ALL_SCENES:
    diagnose_script = REPO_DIR / "utils" / "diagnose_colmap_images.py"
    if diagnose_script.exists():
        log = OUTPUT_DIR / f"{scene.name}_diagnose.log"
        run([sys.executable, diagnose_script, "--scene", scene], cwd=REPO_DIR, log_file=log, check=False)
        print(tail(log, 20))
    else:
        print(f"[{scene.name}] diagnose script not found; skipping COLMAP/image report.")


# =============================================================================
# CELL 4 - Training and rendering helpers
# =============================================================================

def get_gpu_ids():
    try:
        import torch

        n = torch.cuda.device_count()
        return list(range(min(n, MAX_WORKERS)))
    except Exception:
        return [0]


GPU_IDS = get_gpu_ids()
if not GPU_IDS:
    raise RuntimeError("No CUDA GPU visible.")
print(f"Using GPUs: {GPU_IDS}")


def scene_output(scene_path):
    return OUTPUT_DIR / Path(scene_path).name


def checkpoint_iter(path):
    """Read the iteration from local or archived checkpoint names.

    Accepted names include ``chkpnt35000.pth`` and the durable archive form
    ``chkpnt35000_hcm0421.pth``.
    """
    try:
        stem = Path(path).stem
        if not stem.startswith("chkpnt"):
            return -1
        return int(stem[len("chkpnt"):].split("_", 1)[0])
    except (TypeError, ValueError):
        return -1


def is_valid_checkpoint(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            if zf.testzip() is not None:
                return False
        import torch
        # CRC is checked above.  ``meta`` validates the pickle structure and
        # iteration without materialising multi-GB Gaussian tensors in CPU
        # RAM while two scenes are being resumed concurrently.
        payload = torch.load(path, map_location="meta", weights_only=False)
        return (
            isinstance(payload, tuple)
            and len(payload) == 2
            and isinstance(payload[1], int)
            and payload[1] == checkpoint_iter(path)
        )
    except Exception:
        return False


def checkpoint_backup_path(scene_name, iteration):
    """Return the canonical extracted-checkpoint directory name."""
    safe_scene = "".join(
        char.lower() if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(scene_name)
    )
    return CHECKPOINT_DIR / f"chkpnt{iteration}_{safe_scene}"


def is_valid_checkpoint_archive(path):
    """Cheap structural validation for an extracted PyTorch checkpoint."""
    path = Path(path)
    if not path.is_dir():
        return False
    # torch.save writes a zip root such as .chkpnt35000.<random>/data.pkl.
    return any(
        child.is_file() and child.name == "data.pkl" and (child.parent / "data").is_dir()
        for child in path.rglob("data.pkl")
    )


def is_valid_checkpoint_zip(path):
    """Validate the portable zip made from an extracted checkpoint folder."""
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".zip" or path.stat().st_size < 1024:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            return zf.testzip() is None and any(name.endswith("/data.pkl") for name in names)
    except Exception:
        return False


def pack_checkpoint_archive(archive_dir, destination):
    """Recreate a torch-compatible .pth zip from an extracted archive folder."""
    archive_dir, destination = Path(archive_dir), Path(destination)
    if not is_valid_checkpoint_archive(archive_dir):
        return False
    try:
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as zf:
            for entry in archive_dir.rglob("*"):
                if entry.is_file():
                    zf.write(entry, entry.relative_to(archive_dir).as_posix())
        return is_valid_checkpoint(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        return False


def zip_checkpoint_archive(archive_dir, destination):
    """Atomically package an extracted checkpoint directory as a portable zip."""
    archive_dir, destination = Path(archive_dir), Path(destination)
    if not is_valid_checkpoint_archive(archive_dir):
        return None
    # Keep the .zip suffix on the temporary path so the same validator is
    # exercised before atomic publication.
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for entry in archive_dir.rglob("*"):
                if entry.is_file():
                    zf.write(entry, entry.relative_to(archive_dir).as_posix())
        if not is_valid_checkpoint_zip(temporary):
            raise RuntimeError("zip CRC/data.pkl validation failed")
        os.replace(temporary, destination)
        return destination
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        print(f"Checkpoint zip failed for {archive_dir}: {exc}")
        return None


def archive_checkpoint(scene_name, source):
    """Create a verified, portable per-scene checkpoint archive."""
    source = Path(source)
    iteration = checkpoint_iter(source)
    if iteration <= 0 or not is_valid_checkpoint(source):
        print(f"[{scene_name}] refusing to archive invalid checkpoint: {source}")
        return None

    destination = checkpoint_backup_path(scene_name, iteration)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=CHECKPOINT_DIR))
    try:
        with zipfile.ZipFile(source, "r") as zf:
            if zf.testzip() is not None:
                raise RuntimeError("source checkpoint CRC validation failed")
            zf.extractall(temporary)
        if not is_valid_checkpoint_archive(temporary):
            raise RuntimeError("extracted checkpoint has no data.pkl/data payload")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        print(f"[{scene_name}] checkpoint archive failed: {exc}")
        return None

    zip_destination = destination.with_suffix(".zip")
    if CHECKPOINT_ARCHIVE_ZIP:
        if zip_checkpoint_archive(destination, zip_destination) is None:
            # Preserve the extracted, verified archive when zip creation
            # fails; it remains resumable and is safer than losing recovery.
            return destination
        shutil.rmtree(destination, ignore_errors=True)
        destination = zip_destination

    scene_suffix = f"_{checkpoint_backup_path(scene_name, 0).name.split('_', 1)[1]}"
    backups = sorted(
        (
            path for path in CHECKPOINT_DIR.glob(f"chkpnt*{scene_suffix}*")
            if (path.is_dir() or path.suffix.lower() == ".zip")
            and (path.stem if path.suffix.lower() == ".zip" else path.name).endswith(scene_suffix)
        ),
        key=checkpoint_iter,
        reverse=True,
    )
    for stale in backups[CHECKPOINT_BACKUP_KEEP:]:
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
        else:
            stale.unlink(missing_ok=True)
    print(f"[{scene_name}] verified checkpoint backup: {destination}")
    return destination


def restore_archived_checkpoint(scene_name, out_dir, source):
    """Atomically hydrate an archived checkpoint into train.py's model path."""
    source = Path(source)
    destination = Path(out_dir) / f"chkpnt{checkpoint_iter(source)}.pth"
    if is_valid_checkpoint(destination):
        return destination
    # Keep a normal checkpoint-shaped name while validating the temporary
    # archive.  ``is_valid_checkpoint`` derives the expected iteration from
    # the filename, so ``chkpnt40000.pth.restore.tmp`` is rejected despite a
    # valid torch payload (its stem no longer parses as iteration 40000).
    temporary = destination.with_name(
        f"chkpnt{checkpoint_iter(source)}_restore.pth"
    )
    unpacked = None
    try:
        archive_source = source
        if source.suffix.lower() == ".zip":
            if not is_valid_checkpoint_zip(source):
                raise RuntimeError("checkpoint zip failed CRC/data.pkl validation")
            unpacked = Path(tempfile.mkdtemp(prefix=f".{source.stem}.", dir=Path(out_dir)))
            with zipfile.ZipFile(source, "r") as zf:
                zf.extractall(unpacked)
            if not is_valid_checkpoint_archive(unpacked):
                raise RuntimeError("unpacked checkpoint has no data.pkl/data payload")
            archive_source = unpacked
        if not pack_checkpoint_archive(archive_source, temporary):
            raise RuntimeError("repacked checkpoint failed CRC/torch.load validation")
        os.replace(temporary, destination)
        print(f"[{scene_name}] restored verified checkpoint: {source} -> {destination}")
        return destination
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        print(f"[{scene_name}] checkpoint restore failed: {exc}")
        return None
    finally:
        if unpacked is not None:
            shutil.rmtree(unpacked, ignore_errors=True)


def is_valid_ply(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        from plyfile import PlyData

        return len(PlyData.read(path).elements) > 0
    except Exception:
        return False


def latest_checkpoint(out_dir, max_iter=None):
    ckpts = sorted(Path(out_dir).glob("chkpnt*.pth"), key=checkpoint_iter, reverse=True)
    for ckpt in ckpts:
        it = checkpoint_iter(ckpt)
        if max_iter is not None and it > max_iter:
            continue
        if is_valid_checkpoint(ckpt):
            return ckpt
    return None


def latest_archived_checkpoint(scene_path, max_iter=None):
    """Find the most advanced valid per-scene archive, never another scene's."""
    scene_name = Path(scene_path).name
    # Use the same canonical suffix used by checkpoint_backup_path().
    suffix = checkpoint_backup_path(scene_name, 0).name.replace("chkpnt0", "")
    roots = tuple(dict.fromkeys((CHECKPOINT_DIR, CHECKPOINT_INPUT_DIR)))
    candidates = sorted(
        (
            path
            for root in roots if root.is_dir()
            for path in root.glob(f"chkpnt*{suffix}*")
            if (path.is_dir() or path.suffix.lower() == ".zip")
            and (path.stem if path.suffix.lower() == ".zip" else path.name).endswith(suffix)
        ),
        key=checkpoint_iter,
        reverse=True,
    )
    for checkpoint in candidates:
        iteration = checkpoint_iter(checkpoint)
        if max_iter is not None and iteration > max_iter:
            continue
        if is_valid_checkpoint_archive(checkpoint) or is_valid_checkpoint_zip(checkpoint):
            print(f"[{scene_name}] verified archived checkpoint: {checkpoint} (iter {iteration})")
            return checkpoint
    return None


def archived_checkpoint_at(scene_path, iteration):
    """Return a verified archive for exactly ``iteration``, if present."""
    checkpoint = latest_archived_checkpoint(scene_path, max_iter=iteration)
    if checkpoint is not None and checkpoint_iter(checkpoint) == iteration:
        return checkpoint
    return None


def latest_input_checkpoint(scene_path, max_iter=None):
    """Find the newest verified checkpoint shipped with the Kaggle dataset."""
    scene_name = Path(scene_path).name
    candidates = []
    for root in (DATA_ROOT, DATA_ROOT / "private_set1", DATA_ROOT / "checkpoints", DATA_ROOT / "output"):
        if root.exists():
            candidates.extend(root.glob(f"**/{scene_name}/chkpnt*.pth"))
    for ckpt in sorted(set(candidates), key=checkpoint_iter, reverse=True):
        iteration = checkpoint_iter(ckpt)
        if max_iter is not None and iteration > max_iter:
            continue
        if is_valid_checkpoint(ckpt):
            print(f"[{scene_name}] verified input checkpoint: {ckpt} (iter {iteration})")
            return ckpt
    return None


def final_iteration(out_dir):
    pc_dir = Path(out_dir) / "point_cloud"
    iters = []
    if pc_dir.exists():
        for p in pc_dir.glob("iteration_*"):
            try:
                if is_valid_ply(p / "point_cloud.ply"):
                    iters.append(int(p.name.replace("iteration_", "")))
            except ValueError:
                pass
    ckpt = latest_checkpoint(out_dir)
    if ckpt:
        iters.append(checkpoint_iter(ckpt))
    return max(iters) if iters else 0


def ensure_ply_from_checkpoint(out_dir, iteration):
    out_dir = Path(out_dir)
    ply = out_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if is_valid_ply(ply):
        return True

    ckpt = out_dir / f"chkpnt{iteration}.pth"
    if not is_valid_checkpoint(ckpt):
        return False

    helper = OUTPUT_DIR / "extract_checkpoint_ply.py"
    helper.write_text(
        f"""
import argparse
import os
import sys
import torch

sys.path.insert(0, {str(REPO_DIR)!r})
from scene.gaussian_model import GaussianModel

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out_ply", required=True)
parser.add_argument("--sh_degree", type=int, default=3)
args = parser.parse_args()

gaussians = GaussianModel(args.sh_degree, "default")
model_params, _ = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
gaussians.active_sh_degree = model_params[0]
gaussians._xyz = model_params[1]
gaussians._features_dc = model_params[2]
gaussians._features_rest = model_params[3]
gaussians._scaling = model_params[4]
gaussians._rotation = model_params[5]
gaussians._opacity = model_params[6]
os.makedirs(os.path.dirname(args.out_ply), exist_ok=True)
gaussians.save_ply(args.out_ply)
print(f"Extracted {{args.out_ply}}")
""".lstrip(),
        encoding="utf-8",
    )
    rc = run([sys.executable, helper, "--checkpoint", ckpt, "--out_ply", ply], cwd=REPO_DIR, check=False)
    return rc == 0 and is_valid_ply(ply)


def scene_train_config(scene_path):
    scene_name = Path(scene_path).name
    is_closeup = scene_name in CLOSEUP_SCENES
    is_finetune = scene_name in FINETUNE_SCENES
    base_iterations = CLOSEUP_ITERATIONS if is_closeup else ITERATIONS
    if scene_name == "chair" and scene_name in FRESH_SCENES:
        base_iterations = CHAIR_FRESH_ITERATIONS
    cleanup_active = scene_name in CLEANUP_SCENES
    cleanup_steps = CLOSEUP_CLEANUP_STEPS if is_closeup else BTS_CLEANUP_STEPS
    target_iterations = base_iterations + (cleanup_steps if cleanup_active else 0)
    if is_finetune:
        target_iterations = FINETUNE_BASE_ITERATIONS + FINETUNE_STEPS
    target_iterations = int(os.environ.get(
        f"BTS_{scene_name.upper()}_ITERATIONS", str(target_iterations)
    ))
    # A render-only scene must render the exact checked checkpoint, never
    # inherit a group schedule.  This makes the full pinhole profile robust
    # when a pasted config cell accidentally omits BTS_CHAIR_ITERATIONS:
    # bonsai is pinned to its required 80k archive and chair to its 70k one.
    if scene_name in RENDER_ONLY_SCENES and scene_name in REQUIRE_RESUME_SCENES:
        required_iteration = int(os.environ.get(
            f"BTS_{scene_name.upper()}_REQUIRE_RESUME_MIN_ITERATION",
            str(REQUIRE_RESUME_MIN_ITERATION),
        ))
        if required_iteration <= 0:
            raise ValueError(
                f"[{scene_name}] render-only required checkpoint iteration must be positive."
            )
        target_iterations = required_iteration
    if target_iterations <= 0:
        raise ValueError(f"[{scene_name}] BTS_{scene_name.upper()}_ITERATIONS must be positive.")
    cfg = {
        "iterations": target_iterations,
        "resolution": TRAIN_RESOLUTION,
        "densify_grad_threshold": DENSIFY_GRAD_THRESHOLD,
        "densify_until_iter": min(DENSIFY_UNTIL_ITER, target_iterations),
        "percent_dense": PERCENT_DENSE,
        "depth_weight_init": float(os.environ.get("BTS_DEPTH_WEIGHT_INIT", "0.02")),
        "checkpoint_iterations": sorted(set(
            iteration for iteration in CHECKPOINT_ITERATIONS if iteration <= target_iterations
        ) | {target_iterations}),
        "validation_iterations": sorted(set(
            iteration for iteration in VALIDATION_ITERATIONS if iteration <= target_iterations
        ) | {target_iterations}),
        "max_gaussians": MAX_GAUSSIANS,
        "max_new_points_per_densify": MAX_NEW_POINTS_PER_DENSIFY,
        # Spend the memory-safe budget gradually.  Fine wire/background
        # structure gets new splats late, while early training cannot consume
        # all memory in one densification peak.
        "densify_cap_schedule": os.environ.get(
            "BTS_DENSIFY_CAP_SCHEDULE", "10000:1200000,17000:3200000,21000:5200000"
        ).strip(),
        "max_screen_size": int(os.environ.get("BTS_MAX_SCREEN_SIZE", "20")),
        # Do not erase mature thin splats during the extended densification
        # phase.  New high-gradient structure can still split until 30k.
        "opacity_reset_until_iter": int(os.environ.get("BTS_OPACITY_RESET_UNTIL_ITER", "15000")),
        # With no tower masks in the supplied scenes, this is the only loss
        # that explicitly increases gradients on cable/lattice edges.
        "image_edge_loss_weight": float(os.environ.get("BTS_IMAGE_EDGE_LOSS_WEIGHT", "0.02")),
        "position_lr_init": float(os.environ.get("BTS_POSITION_LR_INIT", "0.00016")),
        "position_lr_max_steps": POSITION_LR_MAX_STEPS,
        "alignment_position_lr_scale": float(os.environ.get("BTS_ALIGNMENT_POSITION_LR_SCALE", "1.0")),
        "alignment_feature_lr_scale": float(os.environ.get("BTS_ALIGNMENT_FEATURE_LR_SCALE", "1.0")),
        "alignment_opacity_lr_scale": float(os.environ.get("BTS_ALIGNMENT_OPACITY_LR_SCALE", "1.0")),
        "alignment_scaling_lr_scale": float(os.environ.get("BTS_ALIGNMENT_SCALING_LR_SCALE", "1.0")),
        "alignment_rotation_lr_scale": float(os.environ.get("BTS_ALIGNMENT_ROTATION_LR_SCALE", "1.0")),
        "densify_clone_before_split": False,
        "prune_only_until_iter": 0,
        "prune_only_from_iter": 0,
        "prune_opacity_threshold": float(os.environ.get("BTS_PRUNE_OPACITY_THRESHOLD", "0.005")),
        "prune_min_visibility": 0,
        "prune_warmup_iters": int(os.environ.get("BTS_PRUNE_WARMUP_ITERS", "500")),
        "prune_interval": int(os.environ.get("BTS_PRUNE_INTERVAL", "500")),
        "test_pose_prune_distance": float(os.environ.get("BTS_TEST_POSE_PRUNE_DISTANCE", "0")),
        "test_pose_prune_chunk_size": int(os.environ.get("BTS_TEST_POSE_PRUNE_CHUNK_SIZE", "262144")),
    }
    if is_closeup:
        # bonsai/chair are compact, close-range 360-degree captures.  Smaller
        # split/clone scale and a lower gradient gate create tighter Gaussians;
        # Depth Anything remains a weak anchor instead of flattening fine shape.
        cfg.update({
            "densify_grad_threshold": float(os.environ.get("BTS_CLOSEUP_DENSIFY_GRAD_THRESHOLD", "0.00008")),
            # Do not freeze close-up geometry at 20k: this was the source of
            # the chair/bonsai point-count plateau and blurred close-up edges.
            # The final 10k remains a fixed-geometry convergence phase.
            "densify_until_iter": min(
                int(os.environ.get("BTS_CLOSEUP_DENSIFY_UNTIL_ITER", "30000")),
                target_iterations,
            ),
            "max_gaussians": int(os.environ.get("BTS_CLOSEUP_MAX_GAUSSIANS", str(MAX_GAUSSIANS))),
            "max_new_points_per_densify": int(os.environ.get(
                "BTS_CLOSEUP_MAX_NEW_POINTS_PER_DENSIFY", str(MAX_NEW_POINTS_PER_DENSIFY)
            )),
            # Allocate at least 3M slots by 30k.  The target is reachable only
            # when gradients justify it; it is not an artificial point pad.
            "densify_cap_schedule": os.environ.get(
                "BTS_CLOSEUP_DENSIFY_CAP_SCHEDULE", "10000:1200000,17000:2200000,30000:4000000"
            ).strip(),
            # Small close-up splats must clone even in an interval that also
            # splits larger splats; otherwise foliage/chair detail plateaus.
            "densify_clone_before_split": os.environ.get(
                "BTS_CLOSEUP_CLONE_BEFORE_SPLIT", "1"
            ).strip() != "0",
            # A fresh 40k close-up run should decay its position LR over 40k,
            # independently from the BTS 40k->60k refinement schedule.
            "position_lr_max_steps": int(os.environ.get(
                "BTS_CLOSEUP_POSITION_LR_MAX_STEPS", str(target_iterations)
            )),
            "percent_dense": float(os.environ.get("BTS_CLOSEUP_PERCENT_DENSE", "0.01")),
            "depth_weight_init": float(os.environ.get("BTS_CLOSEUP_DEPTH_WEIGHT_INIT", "0.01")),
            # Keep large, distant window/background splats long enough to
            # converge, while a weak image-edge loss protects chair holes and
            # other high-contrast close-range structure without needing masks.
            "max_screen_size": int(os.environ.get("BTS_CLOSEUP_MAX_SCREEN_SIZE", "64")),
            "opacity_reset_until_iter": int(os.environ.get("BTS_CLOSEUP_OPACITY_RESET_UNTIL_ITER", "12000")),
            "image_edge_loss_weight": float(os.environ.get("BTS_CLOSEUP_IMAGE_EDGE_LOSS_WEIGHT", "0.03")),
            "prune_opacity_threshold": float(os.environ.get(
                "BTS_CLOSEUP_PRUNE_OPACITY_THRESHOLD", "0.005"
            )),
            "prune_min_visibility": int(os.environ.get(
                "BTS_CLOSEUP_PRUNE_MIN_VISIBILITY", "0"
            )),
            "prune_warmup_iters": int(os.environ.get("BTS_CLOSEUP_PRUNE_WARMUP_ITERS", "500")),
            "prune_interval": int(os.environ.get("BTS_CLOSEUP_PRUNE_INTERVAL", "500")),
        })
    if cleanup_active:
        group_prefix = "BTS_CLOSEUP" if is_closeup else "BTS"
        scene_prefix = f"BTS_{scene_name.upper()}"
        cfg.update({
            "prune_only_until_iter": target_iterations,
            "prune_only_from_iter": int(os.environ.get(
                f"{scene_prefix}_CLEANUP_PRUNE_FROM_ITER",
                os.environ.get(f"{group_prefix}_CLEANUP_PRUNE_FROM_ITER", "0"),
            )),
            "prune_opacity_threshold": float(os.environ.get(
                f"{scene_prefix}_CLEANUP_PRUNE_OPACITY_THRESHOLD",
                os.environ.get(
                    f"{group_prefix}_CLEANUP_PRUNE_OPACITY_THRESHOLD",
                    "0.008" if is_closeup else "0.003",
                ),
            )),
            "prune_min_visibility": int(os.environ.get(
                f"{scene_prefix}_CLEANUP_PRUNE_MIN_VISIBILITY",
                os.environ.get(f"{group_prefix}_CLEANUP_PRUNE_MIN_VISIBILITY", "0"),
            )),
            "max_screen_size": int(os.environ.get(
                f"{scene_prefix}_CLEANUP_MAX_SCREEN_SIZE",
                os.environ.get(f"{group_prefix}_CLEANUP_MAX_SCREEN_SIZE", "96" if is_closeup else "256"),
            )),
            "prune_warmup_iters": int(os.environ.get(
                f"{scene_prefix}_CLEANUP_PRUNE_WARMUP_ITERS",
                os.environ.get(f"{group_prefix}_CLEANUP_PRUNE_WARMUP_ITERS", "500" if is_closeup else "1000"),
            )),
            "prune_interval": int(os.environ.get(
                f"{scene_prefix}_CLEANUP_PRUNE_INTERVAL",
                os.environ.get(f"{group_prefix}_CLEANUP_PRUNE_INTERVAL", "750" if is_closeup else "1000"),
            )),
        })
    if is_finetune:
        # Align an existing reconstruction rather than creating another
        # generation of splats.  Starting at 70k makes the normal
        # densification condition false; prune-only then gathers fresh
        # visibility statistics before deleting low-confidence floaters.
        # Disable monocular depth here: it is useful for fresh geometry but
        # can pull an already coherent BTS model toward per-view depth noise.
        group_prefix = "BTS_CLOSEUP" if is_closeup else "BTS"
        scene_prefix = f"BTS_{scene_name.upper()}"
        cfg.update({
            "densify_until_iter": FINETUNE_BASE_ITERATIONS,
            "prune_only_until_iter": target_iterations,
            "prune_only_from_iter": FINETUNE_BASE_ITERATIONS,
            "depth_weight_init": 0.0,
            "position_lr_max_steps": FINETUNE_POSITION_LR_MAX_STEPS,
            "prune_opacity_threshold": float(os.environ.get(
                f"{scene_prefix}_FINETUNE_PRUNE_OPACITY_THRESHOLD",
                os.environ.get(
                    f"{group_prefix}_FINETUNE_PRUNE_OPACITY_THRESHOLD",
                    "0.008" if is_closeup else "0.003",
                ),
            )),
            # A floater which survives only one or two train poses is exactly
            # the noise seen in bonsai's failed novel views.  Require three
            # fresh observations after the 1k-view warm-up for close-up scenes
            # and two for BTS; scene-specific overrides remain available.
            "prune_min_visibility": int(os.environ.get(
                f"{scene_prefix}_FINETUNE_PRUNE_MIN_VISIBILITY",
                os.environ.get(
                    f"{group_prefix}_FINETUNE_PRUNE_MIN_VISIBILITY",
                    "3" if is_closeup else "2",
                ),
            )),
            "max_screen_size": int(os.environ.get(
                f"{scene_prefix}_FINETUNE_MAX_SCREEN_SIZE",
                os.environ.get(f"{group_prefix}_FINETUNE_MAX_SCREEN_SIZE", "96" if is_closeup else "256"),
            )),
            "prune_warmup_iters": int(os.environ.get(
                f"{scene_prefix}_FINETUNE_PRUNE_WARMUP_ITERS",
                os.environ.get(f"{group_prefix}_FINETUNE_PRUNE_WARMUP_ITERS", "1000"),
            )),
            "prune_interval": int(os.environ.get(
                f"{scene_prefix}_FINETUNE_PRUNE_INTERVAL",
                os.environ.get(f"{group_prefix}_FINETUNE_PRUNE_INTERVAL", "500"),
            )),
        })
    scene_prefix = f"BTS_{scene_name.upper()}"
    # A scene-specific cap is useful when capture quality differs sharply.
    # Keep this after the close-up override so, for example, chair can impose
    # a stricter budget than the shared close-up profile.
    cfg["max_gaussians"] = int(os.environ.get(
        f"{scene_prefix}_MAX_GAUSSIANS", str(cfg["max_gaussians"]),
    ))
    cfg["densify_cap_schedule"] = os.environ.get(
        f"{scene_prefix}_DENSIFY_CAP_SCHEDULE", cfg["densify_cap_schedule"],
    ).strip()
    # A mixed recovery run can legitimately contain a resumed detail scene
    # and a fresh scene.  Apply these overrides after the close-up profile so
    # they can use different densification/pruning policies in one queue.
    cfg["densify_until_iter"] = min(int(os.environ.get(
        f"{scene_prefix}_DENSIFY_UNTIL_ITER", str(cfg["densify_until_iter"]),
    )), target_iterations)
    cfg["densify_grad_threshold"] = float(os.environ.get(
        f"{scene_prefix}_DENSIFY_GRAD_THRESHOLD", str(cfg["densify_grad_threshold"]),
    ))
    cfg["max_new_points_per_densify"] = int(os.environ.get(
        f"{scene_prefix}_MAX_NEW_POINTS_PER_DENSIFY", str(cfg["max_new_points_per_densify"]),
    ))
    cfg["percent_dense"] = float(os.environ.get(
        f"{scene_prefix}_PERCENT_DENSE", str(cfg["percent_dense"]),
    ))
    cfg["max_screen_size"] = int(os.environ.get(
        f"{scene_prefix}_MAX_SCREEN_SIZE", str(cfg["max_screen_size"]),
    ))
    cfg["prune_opacity_threshold"] = float(os.environ.get(
        f"{scene_prefix}_PRUNE_OPACITY_THRESHOLD", str(cfg["prune_opacity_threshold"]),
    ))
    cfg["image_edge_loss_weight"] = float(os.environ.get(
        f"{scene_prefix}_IMAGE_EDGE_LOSS_WEIGHT", str(cfg["image_edge_loss_weight"]),
    ))
    cfg["depth_weight_init"] = float(os.environ.get(
        f"{scene_prefix}_DEPTH_WEIGHT_INIT", str(cfg["depth_weight_init"]),
    ))
    cfg["position_lr_init"] = float(os.environ.get(
        f"{scene_prefix}_POSITION_LR_INIT", str(cfg["position_lr_init"]),
    ))
    cfg["position_lr_max_steps"] = int(os.environ.get(
        f"{scene_prefix}_POSITION_LR_MAX_STEPS", str(cfg["position_lr_max_steps"]),
    ))
    for name in ("position", "feature", "opacity", "scaling", "rotation"):
        key = f"alignment_{name}_lr_scale"
        cfg[key] = float(os.environ.get(
            f"{scene_prefix}_ALIGNMENT_{name.upper()}_LR_SCALE", str(cfg[key]),
        ))
    cfg["prune_min_visibility"] = int(os.environ.get(
        f"{scene_prefix}_PRUNE_MIN_VISIBILITY", str(cfg["prune_min_visibility"]),
    ))
    cfg["prune_only_from_iter"] = int(os.environ.get(
        f"{scene_prefix}_PRUNE_ONLY_FROM_ITER", str(cfg["prune_only_from_iter"]),
    ))
    cfg["test_pose_prune_distance"] = float(os.environ.get(
        f"{scene_prefix}_TEST_POSE_PRUNE_DISTANCE",
        str(cfg["test_pose_prune_distance"]),
    ))
    cfg["test_pose_prune_chunk_size"] = int(os.environ.get(
        f"{scene_prefix}_TEST_POSE_PRUNE_CHUNK_SIZE",
        str(cfg["test_pose_prune_chunk_size"]),
    ))
    invalid = (
        cfg["densify_grad_threshold"] <= 0
        or cfg["densify_until_iter"] <= 0
        or not 0 < cfg["percent_dense"] <= 1
        or cfg["depth_weight_init"] < 0
        or cfg["max_gaussians"] < 0
        or cfg["max_new_points_per_densify"] < 0
        or cfg["max_screen_size"] < 0
        or cfg["opacity_reset_until_iter"] < 0
        or cfg["image_edge_loss_weight"] < 0
        or cfg["position_lr_init"] <= 0
        or cfg["position_lr_max_steps"] <= 0
        or any(cfg[f"alignment_{name}_lr_scale"] <= 0 for name in ("position", "feature", "opacity", "scaling", "rotation"))
        or cfg["prune_only_until_iter"] < 0
        or cfg["prune_only_from_iter"] < 0
        or cfg["prune_only_from_iter"] > cfg["prune_only_until_iter"]
        or not 0 < cfg["prune_opacity_threshold"] < 1
        or cfg["prune_min_visibility"] < 0
        or cfg["prune_warmup_iters"] < 0
        or cfg["prune_interval"] <= 0
        or cfg["test_pose_prune_distance"] < 0
        or cfg["test_pose_prune_chunk_size"] <= 0
    )
    if invalid:
        raise ValueError(f"[{scene_name}] invalid scene training configuration: {cfg}")
    return cfg


def fresh_run_marker(scene_name):
    """Marker proving that this scene has already received its one reset."""
    suffix = f"_{FRESH_RUN_ID}" if FRESH_RUN_ID else ""
    return scene_output(Path(scene_name)) / f".fresh_run_started{suffix}"


def scene_is_fresh(scene_name):
    """Whether this invocation must discard an old scene model.

    ``BTS_FRESH_SCENES`` is often kept in a Kaggle config after an interrupted
    session.  Without a durable marker, retrying that session deletes its new
    10k/40k checkpoint and starts over.  Mark immediately after the initial
    reset so subsequent invocations resume normally; delete the marker (or
    the scene output directory) to intentionally force another clean run.
    """
    # The ID changes only the marker namespace; scene selection remains under
    # BTS_FRESH_RUN/BTS_FRESH_SCENES so render-only scenes are never reset.
    requested = FRESH_RUN or scene_name in FRESH_SCENES
    return requested and not fresh_run_marker(scene_name).exists()


def build_train_cmd(scene_path, gpu_id, force_fresh=False):
    scene_name = Path(scene_path).name
    if scene_name in PINHOLE_PREPROCESS_SCENES:
        expected_scene = (PINHOLE_DATA_ROOT / scene_name).resolve()
        actual_scene = Path(scene_path).resolve()
        manifest = actual_scene / ".pinhole_manifest.json"
        if actual_scene != expected_scene or not manifest.is_file():
            raise RuntimeError(
                f"[{scene_name}] PINHOLE REQUIRED but training source is {actual_scene}; "
                f"expected {expected_scene} with .pinhole_manifest.json. "
                "Use the latest kaggle_notebook.py and restart from the configuration cell."
            )
        print(f"[{scene_name}] PINHOLE TRAINING CONFIRMED: {actual_scene}")
    out_dir = scene_output(scene_path)
    cfg = scene_train_config(scene_path)
    print(
        f"[{scene_name}] profile={'closeup' if scene_name in CLOSEUP_SCENES else 'bts'} "
        f"| dense={cfg['percent_dense']} | grad={cfg['densify_grad_threshold']} "
        f"| depth_weight={cfg['depth_weight_init']} | screen_prune={cfg['max_screen_size']} "
        f"| cap_schedule={cfg['densify_cap_schedule'] or 'off'} | max_new={cfg['max_new_points_per_densify']} "
        f"| edge_loss={cfg['image_edge_loss_weight']} | prune_until={cfg['prune_only_until_iter']} "
        f"| prune_from={cfg['prune_only_from_iter']} "
        f"| prune_opacity={cfg['prune_opacity_threshold']} | prune_warmup={cfg['prune_warmup_iters']} "
        f"| prune_visibility={cfg['prune_min_visibility']} | prune_interval={cfg['prune_interval']} "
        f"| align_lr=xyz:{cfg['alignment_position_lr_scale']},feat:{cfg['alignment_feature_lr_scale']},"
        f"opacity:{cfg['alignment_opacity_lr_scale']},scale:{cfg['alignment_scaling_lr_scale']},"
        f"rot:{cfg['alignment_rotation_lr_scale']} "
        f"| checkpoints={cfg['checkpoint_iterations']} "
        f"| test_pose_prune={cfg['test_pose_prune_distance']}/{cfg['test_pose_prune_chunk_size']} "
        f"| validate/render={cfg['validation_iterations']}"
    )
    resume = None
    target_iterations = cfg["iterations"]
    if RESUME_LOCAL and not force_fresh and not scene_is_fresh(scene_name):
        # A SIGKILL may leave the scene output partially cleaned or corrupted.
        # Choose the newest valid copy across the live model directory and the
        # independently verified archive.
        local_resume = latest_checkpoint(out_dir, max_iter=target_iterations - 1)
        archived_resume = latest_archived_checkpoint(scene_path, max_iter=target_iterations - 1)
        candidates = [path for path in (local_resume, archived_resume) if path is not None]
        local_resume = max(candidates, key=checkpoint_iter) if candidates else None
    else:
        local_resume = None
    input_resume = (
        latest_input_checkpoint(scene_path, max_iter=target_iterations - 1)
        if RESUME_INPUT and not force_fresh else None
    )
    # Input is the canonical 50k baseline.  A strictly newer local checkpoint
    # can only come from an interrupted continuation, so retain it safely.
    # At equal iteration prefer the attached input archive over stale working
    # files from a previous profile.
    if input_resume is not None and (
        local_resume is None or checkpoint_iter(input_resume) >= checkpoint_iter(local_resume)
    ):
        resume = input_resume
        resume_source = "input"
    elif local_resume is not None:
        resume = local_resume
        resume_source = "local"
    else:
        resume = None
        resume_source = None
    if resume is not None:
        print(f"[{scene_name}] resuming verified {resume_source} checkpoint: {resume}")
    elif RESUME_INPUT and not force_fresh:
        print(f"[{scene_name}] no verified input or local checkpoint found; starting clean.")
    else:
        print(f"[{scene_name}] fresh policy active; starting without a checkpoint.")

    if scene_name in REQUIRE_RESUME_SCENES:
        required_resume_iteration = int(os.environ.get(
            f"BTS_{scene_name.upper()}_REQUIRE_RESUME_MIN_ITERATION",
            str(REQUIRE_RESUME_MIN_ITERATION),
        ))
        if required_resume_iteration < 0:
            raise ValueError(f"[{scene_name}] required resume iteration must be non-negative.")
        if resume is None:
            raise RuntimeError(
                f"[{scene_name}] requires a verified recovery checkpoint at iteration >= "
                f"{required_resume_iteration}; refusing a fresh start."
            )
        if checkpoint_iter(resume) < required_resume_iteration:
            raise RuntimeError(
                f"[{scene_name}] checkpoint is step {checkpoint_iter(resume)}, but recovery requires "
                f">= {required_resume_iteration}; refusing an underspecified resume."
            )

    if scene_name in FINETUNE_SCENES:
        if resume is None:
            raise RuntimeError(
                f"[{scene_name}] alignment fine-tune requires a verified checkpoint at "
                f"iteration >= {FINETUNE_BASE_ITERATIONS}; refusing a fresh run."
            )
        if checkpoint_iter(resume) < FINETUNE_BASE_ITERATIONS:
            raise RuntimeError(
                f"[{scene_name}] latest checkpoint is step {checkpoint_iter(resume)}, but alignment "
                f"requires >= {FINETUNE_BASE_ITERATIONS}; refusing an underspecified resume."
            )

    cmd = [
        sys.executable,
        REPO_DIR / "train.py",
        "-s",
        scene_path,
        "-m",
        out_dir,
        "-r",
        str(cfg["resolution"]),
        "--sh_degree",
        str(SH_DEGREE),
        "--data_device",
        "cpu",
        "--iterations",
        str(target_iterations),
        "--lambda_dssim",
        "0.2",
        "--position_lr_init",
        str(cfg["position_lr_init"]),
        "--position_lr_max_steps",
        str(cfg["position_lr_max_steps"]),
        "--alignment_position_lr_scale",
        str(cfg["alignment_position_lr_scale"]),
        "--alignment_feature_lr_scale",
        str(cfg["alignment_feature_lr_scale"]),
        "--alignment_opacity_lr_scale",
        str(cfg["alignment_opacity_lr_scale"]),
        "--alignment_scaling_lr_scale",
        str(cfg["alignment_scaling_lr_scale"]),
        "--alignment_rotation_lr_scale",
        str(cfg["alignment_rotation_lr_scale"]),
        "--densification_interval",
        "100",
        "--densify_grad_threshold",
        str(cfg["densify_grad_threshold"]),
        "--densify_until_iter",
        str(cfg["densify_until_iter"]),
        "--densify_cap_schedule",
        cfg["densify_cap_schedule"],
        "--max_new_points_per_densify",
        str(cfg["max_new_points_per_densify"]),
        "--prune_only_until_iter",
        str(cfg["prune_only_until_iter"]),
        "--prune_only_from_iter",
        str(cfg["prune_only_from_iter"]),
        "--prune_opacity_threshold",
        str(cfg["prune_opacity_threshold"]),
        "--prune_min_visibility",
        str(cfg["prune_min_visibility"]),
        "--prune_warmup_iters",
        str(cfg["prune_warmup_iters"]),
        "--prune_interval",
        str(cfg["prune_interval"]),
        "--test_pose_prune_distance",
        str(cfg["test_pose_prune_distance"]),
        "--test_pose_prune_chunk_size",
        str(cfg["test_pose_prune_chunk_size"]),
        "--percent_dense",
        str(cfg["percent_dense"]),
        "--opacity_reset_interval",
        "3000",
        "--opacity_reset_until_iter",
        str(cfg["opacity_reset_until_iter"]),
        "--max_screen_size",
        str(cfg["max_screen_size"]),
        "--image_edge_loss_weight",
        str(cfg["image_edge_loss_weight"]),
        "--min_free_disk_gb",
        str(MIN_FREE_DISK_GB),
        "--disk_check_interval",
        str(DISK_CHECK_INTERVAL),
        "--depth_weight_init",
        str(cfg["depth_weight_init"]),
        "--test_iterations",
        "-1",
        "--validation_iterations",
        *[str(x) for x in cfg["validation_iterations"]],
        "--validation_fraction",
        str(VALIDATION_FRACTION),
        "--checkpoint_iterations",
        *[str(x) for x in cfg["checkpoint_iterations"]],
        "--save_iterations",
        str(target_iterations),
        "--disable_viewer",
        # Do not pass --quiet here. train.py sends it to safe_state(), whose
        # stdout wrapper drops every print and tqdm update. With subprocess
        # output streamed this otherwise makes healthy GPU workers look hung
        # while the notebook waits in as_completed().
        "--progress_name",
        f"{scene_name}-gpu{gpu_id}",
        *optional_depth_args(scene_path),
        *optional_mask_args(scene_path),
    ]
    if VALIDATION_HOLDOUT:
        cmd.append("--validation_holdout")
    if VALIDATION_LPIPS_FINAL:
        cmd.append("--validation_lpips_final")
    if cfg["test_pose_prune_distance"] <= 0:
        # Avoid allocating black placeholder images for test views unless the
        # training profile explicitly needs their poses for pruning.
        cmd.append("--skip_test_poses")
    if cfg["densify_clone_before_split"]:
        cmd.append("--densify_clone_before_split")

    if SUPPORTS_MAX_GAUSSIANS:
        cmd.extend(["--max_gaussians", str(cfg["max_gaussians"])])
    if USE_ANTIALIASING:
        cmd.append("--antialiasing")

    cmd.extend(["--stop_at_unix_time", str(TRAIN_STOP_AT_UNIX_TIME)])
    if gpu_id:
        cmd.extend(["--checkpoint_stagger_seconds", str(gpu_id * CHECKPOINT_STAGGER_SECONDS)])

    if resume:
        cmd.extend(["--start_checkpoint", resume])
    if USE_WANDB:
        cmd.extend(["--use_wandb", "--wandb_project", WANDB_PROJECT])
        if WANDB_ENTITY:
            cmd.extend(["--wandb_entity", WANDB_ENTITY])
        # Pass the run name explicitly so train.py uses it directly in wandb.init().
        # Relying on the WANDB_NAME env-var alone is unreliable: the SDK prioritises
        # the name= keyword argument inside wandb.init(), so without this flag the
        # env-var is silently ignored.
        cmd.extend(["--wandb_name", f"{scene_name}-gpu{gpu_id}"])
        cmd.extend(["--wandb_log_interval", str(WANDB_LOG_INTERVAL)])

    env = {
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "WANDB_API_KEY": WANDB_API_KEY,
        "WANDB_MODE": "online",
        "WANDB_NAME": f"{scene_name}-gpu{gpu_id}",
        "WANDB_LOG_FILE": str(OUTPUT_DIR / f"{scene_name}_train.log"),
        # Stagger WandB service initialisation: GPU-0 inits first, GPU-1 waits
        # 15 s so the wandb-service daemon is already running when the second
        # process starts.  This prevents the silent "only one run appears" bug
        # caused by two processes racing to spawn the shared wandb-service.
        "WANDB_INIT_TIMEOUT": "120",
        "WANDB__SERVICE_WAIT": str(max(300, int(gpu_id * 15 + 300))),
    }
    return cmd, env


def build_render_cmd(scene_path, gpu_id, iteration):
    out_dir = scene_output(scene_path)
    scene_name = Path(scene_path).name
    ensemble_scales = CLOSEUP_RENDER_ENSEMBLE_SCALES if scene_name in CLOSEUP_SCENES else RENDER_ENSEMBLE_SCALES
    cmd = [
        sys.executable,
        REPO_DIR / "render.py",
        "-s",
        scene_path,
        "-m",
        out_dir,
        "-r",
        str(RENDER_RESOLUTION),
        "--skip_train",
        "--iteration",
        str(iteration),
        "--sh_degree",
        str(SH_DEGREE),
        "--quiet",
        "--ensemble_scales",
        *[str(scale) for scale in ensemble_scales],
    ]
    if USE_ANTIALIASING:
        cmd.append("--antialiasing")
    scene_prefix = f"BTS_{scene_name.upper()}"
    near_distance = float(os.environ.get(
        f"{scene_prefix}_RENDER_NEAR_CAMERA_DISTANCE",
        os.environ.get("BTS_RENDER_NEAR_CAMERA_DISTANCE", "0"),
    ))
    scale_to_distance = float(os.environ.get(
        f"{scene_prefix}_RENDER_NEAR_CAMERA_SCALE_TO_DISTANCE",
        os.environ.get("BTS_RENDER_NEAR_CAMERA_SCALE_TO_DISTANCE", "0"),
    ))
    sharpen_amount = float(os.environ.get(
        f"{scene_prefix}_RENDER_SHARPEN_AMOUNT",
        os.environ.get(
            "BTS_CLOSEUP_RENDER_SHARPEN_AMOUNT" if scene_name in CLOSEUP_SCENES else "BTS_RENDER_SHARPEN_AMOUNT",
            "0",
        ),
    ))
    if near_distance < 0.0 or scale_to_distance < 0.0 or sharpen_amount < 0.0:
        raise ValueError(f"[{scene_name}] render cull/sharpen values must be non-negative")
    if near_distance or scale_to_distance:
        cmd.extend([
            "--near_camera_distance", str(near_distance),
            "--near_camera_scale_to_distance", str(scale_to_distance),
        ])
    if sharpen_amount:
        cmd.extend(["--sharpen_amount", str(sharpen_amount)])
    # Semicolon separates rules so image file names remain unmodified.  This
    # is intentionally an inference-only exception, scoped to a known bad
    # test pose rather than a global mutation of the trained model.
    cull_rules = os.environ.get(
        f"BTS_{scene_name.upper()}_RENDER_NEAR_CAMERA_CULLS",
        os.environ.get("BTS_RENDER_NEAR_CAMERA_CULLS", ""),
    ).strip()
    if cull_rules:
        for rule in cull_rules.split(";"):
            rule = rule.strip()
            if rule:
                cmd.extend(["--near_camera_cull", rule])
    return cmd, {"CUDA_VISIBLE_DEVICES": str(gpu_id)}


def cleanup_intermediate(out_dir, keep_iter):
    out_dir = Path(out_dir)
    for event in out_dir.glob("events.out.tfevents*"):
        event.unlink(missing_ok=True)

    pc_dir = out_dir / "point_cloud"
    if pc_dir.exists():
        for p in pc_dir.glob("iteration_*"):
            if p.name != f"iteration_{keep_iter}":
                shutil.rmtree(p, ignore_errors=True)

    for ckpt in out_dir.glob("chkpnt*.pth"):
        if checkpoint_iter(ckpt) < keep_iter:
            ckpt.unlink(missing_ok=True)


def cleanup_scene_output_after_submission(out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)


def copy_renders_to_submission(scene_path, iteration):
    scene_name = Path(scene_path).name
    out_dir = scene_output(scene_path)
    render_dir = out_dir / "test" / f"ours_{iteration}" / "renders"
    dest = SUBMISSION_DIR / scene_name
    if not render_dir.exists():
        print(f"[{scene_name}] missing render dir: {render_dir}")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    # A rerun must replace partial/stale renders rather than append to them.
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]:
        for old_image in dest.glob(ext):
            old_image.unlink(missing_ok=True)
    images = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]:
        images.extend(sorted(render_dir.glob(ext)))

    for img in _tqdm(images, desc=f"[{scene_name}] copying renders", unit="img", leave=False):
        shutil.copy2(img, dest / img.name)
    print(f"[{scene_name}] copied {len(images)} renders to {dest}")
    return len(images)


def submission_image_count(scene_name):
    dest = SUBMISSION_DIR / scene_name
    if not dest.exists():
        return 0
    total = 0
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]:
        total += len(list(dest.glob(ext)))
    return total


def expected_submission_names(scene_path):
    """Return exactly the valid image names requested by test_poses.csv."""
    poses = Path(scene_path) / "test" / "test_poses.csv"
    if not poses.exists():
        return set()
    names = set()
    for line in poses.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 14 and fields[0]:
            names.add(fields[0])
    return names


def submission_names(scene_name):
    dest = SUBMISSION_DIR / scene_name
    if not dest.exists():
        return set()
    return {
        p.name for p in dest.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }


def train_and_render_scene(scene_path, gpu_id):
    scene_path = Path(scene_path)
    scene_name = scene_path.name
    target_iterations = scene_train_config(scene_path)["iterations"]
    fresh_scene = scene_is_fresh(scene_name)
    out_dir = scene_output(scene_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{scene_name}] worker started on GPU {gpu_id}; running preflight.", flush=True)

    if fresh_scene:
        # This is intentionally scoped to exactly one selected scene.  It runs
        # before checkpoint discovery so --start_checkpoint can never revive a
        # model from the previous experiment.
        if out_dir.exists():
            shutil.rmtree(out_dir)
        scene_submission = SUBMISSION_DIR / scene_name
        if scene_submission.exists():
            shutil.rmtree(scene_submission)
        out_dir.mkdir(parents=True, exist_ok=True)
        fresh_run_marker(scene_name).touch()
        print(f"[{scene_name}] fresh-scene policy: cleared prior model and renders.")

    if RESUME_LOCAL and not fresh_scene:
        # Hydrate a backup before the final-artifact check as well: an archived
        # 40k checkpoint can be rendered immediately without retraining.
        # Render-only jobs require the exact requested model, rather than a
        # newer cleanup/refinement checkpoint with a different schedule.
        archived = (
            archived_checkpoint_at(scene_path, target_iterations)
            if scene_name in RENDER_ONLY_SCENES
            else latest_archived_checkpoint(scene_path)
        )
        local = latest_checkpoint(out_dir)
        if archived is not None and (
            local is None
            or (scene_name in RENDER_ONLY_SCENES and checkpoint_iter(local) != target_iterations)
            or checkpoint_iter(archived) > checkpoint_iter(local)
        ):
            restore_archived_checkpoint(scene_name, out_dir, archived)

    # A complete submission from an earlier (for example 40k) experiment must
    # not prevent an explicit higher-iteration refinement run from resuming.
    # The old behaviour returned here solely because images existed, even when
    # BTS_ITERATIONS had been raised and no model at that target existed.
    final_ply = out_dir / "point_cloud" / f"iteration_{target_iterations}" / "point_cloud.ply"
    final_ckpt = out_dir / f"chkpnt{target_iterations}.pth"
    target_model_exists = is_valid_ply(final_ply) or is_valid_checkpoint(final_ckpt)
    expected_names = expected_submission_names(scene_path)
    existing_names = submission_names(scene_name)
    if expected_names and existing_names == expected_names and target_model_exists:
        print(f"[{scene_name}] submission already has all {len(existing_names)} expected images, skipping train/render.")
        return scene_name, 0
    if expected_names and existing_names == expected_names:
        print(
            f"[{scene_name}] existing submission is from an earlier iteration; "
            f"target {target_iterations} has no verified model, so refinement will resume."
        )
    if existing_names and existing_names != expected_names:
        print(f"[{scene_name}] incomplete/stale submission ({len(existing_names)}/{len(expected_names)}); rerendering.")

    if hours_remaining() < KAGGLE_STOP_BUFFER_MIN / 60:
        print(f"[{scene_name}] time budget exhausted, skipping.")
        return scene_name, 2

    free_gb, total_gb = disk_free_gb()
    if free_gb < MIN_FREE_DISK_GB:
        print(f"[{scene_name}] disk free {free_gb:.1f}GB < {MIN_FREE_DISK_GB:.1f}GB, skipping new training to preserve verified checkpoints.")
        return scene_name, 5

    print(f"[{scene_name}] checking final model artifacts.", flush=True)
    if is_valid_ply(final_ply) or is_valid_checkpoint(final_ckpt):
        if is_valid_checkpoint(final_ckpt):
            archive_checkpoint(scene_name, final_ckpt)
        print(f"[{scene_name}] final model exists, skipping training.")
    elif scene_name in RENDER_ONLY_SCENES:
        print(
            f"[{scene_name}] render-only policy: missing verified iteration "
            f"{target_iterations} checkpoint/PLY; refusing to train."
        )
        return scene_name, 8
    else:
        print(f"[{scene_name}] preparing train.py command.", flush=True)
        cmd, env = build_train_cmd(scene_path, gpu_id, force_fresh=fresh_scene)
        log = OUTPUT_DIR / f"{scene_name}_train.log"
        print(f"[{scene_name}] train on GPU {gpu_id} | images={count_images(scene_path)} | log={log}")
        # Stagger the WandB-service startup: GPU-0 initialises it first, and
        # subsequent workers wait long enough for the daemon to be ready before
        # they call wandb.init().  Without this delay the two processes race to
        # spawn the shared wandb-service socket and the loser fails silently,
        # which is why only one scene appeared in the WandB dashboard.
        if gpu_id and USE_WANDB:
            import time as _time
            _wandb_stagger = 15 * gpu_id
            print(f"[{scene_name}] Waiting {_wandb_stagger}s for WandB service to start on GPU 0...")
            _time.sleep(_wandb_stagger)
        rc = run(cmd, cwd=REPO_DIR, env=env, log_file=log, check=False, stream=True)

        # train.py publishes checkpoints atomically.  Synchronise the newest
        # one even when it exited with -9, so the next Kaggle run resumes the
        # last completed step rather than silently starting from zero.
        newest = latest_checkpoint(out_dir)
        if newest is not None:
            archive_checkpoint(scene_name, newest)

        if rc != 0:
            print(f"[{scene_name}] training failed rc={rc}")
            print(tail(log, 80))
            return scene_name, rc

    # Render the requested schedule point exactly.  A prior experiment can
    # leave a later valid checkpoint in this output directory; choosing the
    # numerically latest artifact would silently submit that other experiment.
    it = target_iterations
    if not (is_valid_ply(final_ply) or is_valid_checkpoint(final_ckpt)):
        completed_iter = final_iteration(out_dir)
        print(
            f"[{scene_name}] training stopped at iter {completed_iter}; expected {target_iterations}. "
            "Rendering is deferred until a verified final checkpoint/PLY exists."
        )
        return scene_name, 6

    if not ensure_ply_from_checkpoint(out_dir, it):
        ply = out_dir / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
        print(f"[{scene_name}] no PLY found after training: {ply}")
        return scene_name, 3

    cmd, env = build_render_cmd(scene_path, gpu_id, it)
    log = OUTPUT_DIR / f"{scene_name}_render.log"
    print(f"[{scene_name}] render iteration {it} on GPU {gpu_id} | log={log}")
    rc = run(cmd, cwd=REPO_DIR, env=env, log_file=log, check=False, stream=True)
    if rc != 0:
        print(f"[{scene_name}] render failed rc={rc}")
        print(tail(log, 80))
        return scene_name, rc

    n = copy_renders_to_submission(scene_path, it)
    if n <= 0:
        return scene_name, 4

    actual_names = submission_names(scene_name)
    if expected_names and actual_names != expected_names:
        print(
            f"[{scene_name}] render set is incomplete; preserving model for retry "
            f"(expected={len(expected_names)}, got={len(actual_names)}, "
            f"missing={len(expected_names - actual_names)}, "
            f"extra={len(actual_names - expected_names)})."
        )
        return scene_name, 7

    # Only reclaim the large model artifacts after the copied render set has
    # passed the same exact-name contract that Cell 6 enforces at packaging.
    keep_scene_artifacts = KEEP_MODEL_ARTIFACTS and (
        not KEEP_MODEL_SCENES or scene_name in KEEP_MODEL_SCENES
    )
    if keep_scene_artifacts:
        cleanup_intermediate(out_dir, it)
        print(f"[{scene_name}] retaining final model artifacts (BTS_KEEP_MODEL_ARTIFACTS=1).")
    else:
        cleanup_intermediate(out_dir, it)
        cleanup_scene_output_after_submission(out_dir)
    free_gb, total_gb = disk_free_gb()
    print(f"[{scene_name}] done. Disk: {free_gb:.1f}/{total_gb:.1f} GB free")
    return scene_name, 0


# =============================================================================
# CELL 5 - Run two-phase GPU queue
# =============================================================================

def scene_priority(scene_path):
    out_dir = scene_output(scene_path)
    scene_name = Path(scene_path).name
    target_iterations = scene_train_config(scene_path)["iterations"]
    final_ply = out_dir / "point_cloud" / f"iteration_{target_iterations}" / "point_cloud.ply"
    if is_valid_ply(final_ply) or is_valid_checkpoint(out_dir / f"chkpnt{target_iterations}.pth"):
        return (3, 0)
    partial = None
    if RESUME_LOCAL and not scene_is_fresh(scene_name):
        candidates = [
            checkpoint for checkpoint in (
                latest_checkpoint(out_dir, max_iter=target_iterations - 1),
                latest_archived_checkpoint(scene_path, max_iter=target_iterations - 1),
            )
            if checkpoint is not None
        ]
        partial = max(candidates, key=checkpoint_iter) if candidates else None
    if partial:
        # Finish the most advanced resumable models first.
        return (2, checkpoint_iter(partial))
    # BTS is the submission priority.  Starting bonsai/chair on both GPUs
    # previously pushed all five tower scenes into the global deadline.
    return (1 if Path(scene_path).name not in CLOSEUP_SCENES else 0, 0)


def foreground_mask_coverage(scene_path):
    """Return the best matching mask directory and its image-stem coverage."""
    root = train_root(scene_path)
    image_stems = {path.stem for path in (root / "images").glob("*") if path.is_file()}
    if not image_stems:
        return None, 0, 0
    best_name = None
    best_count = 0
    for name in ("foreground_masks", "masks", "mask", "foreground"):
        mask_root = root / name
        if not mask_root.is_dir():
            continue
        mask_stems = {path.stem for path in mask_root.glob("*") if path.is_file()}
        matched = len(image_stems & mask_stems)
        if matched > best_count:
            best_name, best_count = name, matched
    return best_name, best_count, len(image_stems)


def preflight_submission_contract(scenes):
    """Fail before GPU work when a quality profile's required inputs are absent."""
    failures = []
    for scene in scenes:
        scene_name = scene.name
        root = train_root(scene)
        image_count = count_images(scene)
        if image_count == 0:
            failures.append(f"{scene_name}: train/images has no supported image files")
        if not (scene / "test" / "test_poses.csv").is_file():
            failures.append(f"{scene_name}: missing test/test_poses.csv")

        if scene_name in REQUIRE_FOREGROUND_MASK_SCENES:
            mask_name, matched, total = foreground_mask_coverage(scene)
            coverage = matched / max(1, total)
            if mask_name is None or coverage < MIN_FOREGROUND_MASK_COVERAGE:
                failures.append(
                    f"{scene_name}: foreground masks cover {matched}/{total} ({coverage:.1%}); "
                    f"need at least {MIN_FOREGROUND_MASK_COVERAGE:.1%}"
                )
            else:
                print(
                    f"[{scene_name}] preflight masks: {mask_name} {matched}/{total} "
                    f"({coverage:.1%}) meets required coverage."
                )

        if scene_name in REQUIRE_CHECKPOINT_ARCHIVE_SCENES:
            target = scene_train_config(scene)["iterations"]
            archive = archived_checkpoint_at(scene, target)
            if archive is None:
                failures.append(
                    f"{scene_name}: missing verified archive chkpnt{target}_{scene_name.lower()} "
                    f"(.zip or extracted) under BTS_CHECKPOINT_INPUT_DIR={CHECKPOINT_INPUT_DIR}"
                )
            else:
                print(f"[{scene_name}] preflight checkpoint: {archive}")
    if failures:
        raise RuntimeError("Submission input preflight failed:\n- " + "\n- ".join(failures))


preflight_submission_contract(ALL_SCENES)


ALL_SCENES = sorted(ALL_SCENES, key=scene_priority, reverse=True)
first_phase_scenes = [scene for scene in ALL_SCENES if scene.name in TRAIN_FIRST_SCENES]
second_phase_scenes = [scene for scene in ALL_SCENES if scene.name not in TRAIN_FIRST_SCENES]
concurrent_priority_scenes = first_phase_scenes + second_phase_scenes


print("=" * 80)
print(
    f"Starting pipeline: {len(ALL_SCENES)} scenes, GPUs={GPU_IDS}, BTS iterations={ITERATIONS}, "
    f"close-up iterations={CLOSEUP_ITERATIONS}, fresh={sorted(FRESH_SCENES)}, "
    f"train-first={[scene.name for scene in first_phase_scenes]}, "
    f"train-first-exclusive={TRAIN_FIRST_EXCLUSIVE}, "
    f"render-only={sorted(RENDER_ONLY_SCENES)}, checkpoints={CHECKPOINT_ITERATIONS}, "
    f"validation/render={VALIDATION_ITERATIONS}"
)
print("=" * 80)

_scene_start_times: dict = {}

results = []
_pipeline_bar = _tqdm(total=len(ALL_SCENES), desc="Scenes", unit="scene", dynamic_ncols=True)
active_futures = {}


def run_phase(scenes, phase_name):
    """Run one queue phase and return each scene's result.

    Submission order is preserved: when priority scenes are prepended they
    acquire the first GPU(s), while later scenes fill otherwise idle GPUs.
    """
    global active_futures
    if not scenes:
        return []
    print(f"Starting {phase_name}: {[scene.name for scene in scenes]}")
    gpu_queue = queue.Queue()
    # Reserve GPUs for the first submitted scenes.  This makes a priority
    # chair task deterministically acquire GPU 0 instead of relying on thread
    # scheduling luck, while remaining scenes wait for one of those GPUs.
    reserved_gpus = GPU_IDS[:min(len(GPU_IDS), len(scenes))]
    for gpu in GPU_IDS[len(reserved_gpus):]:
        gpu_queue.put(gpu)

    def worker(scene, reserved_gpu=None):
        gpu = reserved_gpu if reserved_gpu is not None else gpu_queue.get()
        try:
            print(f"[{Path(scene).name}] acquired GPU {gpu} from {phase_name} queue.", flush=True)
            return train_and_render_scene(scene, gpu)
        finally:
            gpu_queue.put(gpu)

    phase_results = []
    executor = ThreadPoolExecutor(max_workers=len(GPU_IDS))
    active_futures = {
        executor.submit(worker, scene, reserved_gpus[index] if index < len(reserved_gpus) else None): scene
        for index, scene in enumerate(scenes)
    }
    try:
        for future in as_completed(active_futures):
            scene = active_futures[future]
            scene_name = Path(scene).name
            try:
                result = future.result()
            except Exception as exc:
                result = (scene_name, 99)
                print(f"[{scene_name}] unhandled error: {exc}")
            phase_results.append(result)
            rc = result[1]
            rc_label = {0: "OK", 2: "timeout", 3: "no-PLY", 4: "no-renders",
                        5: "low-disk", 6: "partial", 7: "invalid-render-set",
                        8: "missing-render-checkpoint", 99: "exception"}.get(rc, f"rc={rc}")
            free_gb, _ = disk_free_gb()
            n_imgs = submission_image_count(scene_name)
            _pipeline_bar.set_postfix(scene=scene_name, status=rc_label, imgs=n_imgs,
                                      disk_free=f"{free_gb:.1f}GB")
            _pipeline_bar.update(1)
            print(f"Completed {phase_name}: {result} | imgs={n_imgs} | disk_free={free_gb:.1f}GB")
    except BaseException:
        for future in active_futures:
            future.cancel()
        stop_active_processes()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    finally:
        active_futures = {}
    return phase_results


try:
    if TRAIN_FIRST_EXCLUSIVE:
        first_phase_results = run_phase(first_phase_scenes, "phase 1: priority training")
        results.extend(first_phase_results)
        failed_first_phase = [name for name, rc in first_phase_results if rc != 0]
        if failed_first_phase:
            raise RuntimeError(
                "Phase 2 is blocked because train-first scenes did not finish: "
                f"{failed_first_phase}. Submission packaging is intentionally skipped."
            )
        results.extend(run_phase(second_phase_scenes, "phase 2: remaining scenes"))
    else:
        # Chair is submitted first and therefore receives GPU 0, while GPU 1
        # immediately starts the next fine-tune instead of waiting ~70k steps.
        results.extend(run_phase(concurrent_priority_scenes, "concurrent priority queue"))
except KeyboardInterrupt:
    # Do not let ThreadPoolExecutor.__exit__ wait indefinitely for subprocesses
    # after an interrupted Kaggle cell.  Stop active children first, then join
    # the short-lived reader/worker threads.
    print("KeyboardInterrupt: stopping active train/render subprocesses...")
    for future in active_futures:
        future.cancel()
    stop_active_processes()
    raise
except BaseException:
    stop_active_processes()
    raise
finally:
    _pipeline_bar.close()

print("Pipeline results:", results)

# WandB: log final pipeline summary table
if USE_WANDB:
    try:
        import wandb as _wandb
        _rows = [[name, rc, {0:"OK",2:"timeout",3:"no-PLY",4:"no-renders",
                              5:"low-disk",6:"partial",7:"invalid-render-set",
                              8:"missing-render-checkpoint",99:"exception"}.get(rc, str(rc)),
                  submission_image_count(name)]
                 for name, rc in results]
        _wandb.log({"pipeline/summary": _wandb.Table(
            columns=["scene", "rc", "status", "rendered_images"],
            data=_rows,
        )})
    except Exception as _e:
        print(f"WandB summary log failed: {_e}")


# =============================================================================
# CELL 6 - Package submission.zip
# =============================================================================

def collect_submission_images():
    pairs = []
    for root, _, files in os.walk(SUBMISSION_DIR):
        for name in sorted(files):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                full = Path(root) / name
                arcname = full.relative_to(SUBMISSION_DIR).as_posix()
                pairs.append((full, arcname))
    return pairs


def pack_lossless(pairs):
    if SUBMISSION_ZIP.exists():
        SUBMISSION_ZIP.unlink()
    with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for full, arcname in _tqdm(pairs, desc="Packing (lossless)", unit="img"):
            zf.write(full, arcname)
    return SUBMISSION_ZIP.stat().st_size


def pack_as_jpeg(pairs, quality):
    from PIL import Image

    if SUBMISSION_ZIP.exists():
        SUBMISSION_ZIP.unlink()
    with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for full, arcname in _tqdm(pairs, desc=f"Packing JPEG q={quality}", unit="img"):
            try:
                img = Image.open(full).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality, optimize=True, subsampling=0)
                zf.writestr(arcname, buf.getvalue())
            except Exception:
                zf.write(full, arcname)
    return SUBMISSION_ZIP.stat().st_size


pairs = collect_submission_images()
missing = []
invalid_submission = []
for scene in ALL_SCENES:
    scene_dir = SUBMISSION_DIR / scene.name
    expected_names = expected_submission_names(scene)
    actual_names = submission_names(scene.name)
    if not scene_dir.exists() or not actual_names:
        missing.append(scene.name)
    elif expected_names and actual_names != expected_names:
        invalid_submission.append(
            f"{scene.name}: expected {len(expected_names)}, got {len(actual_names)} "
            f"(missing={len(expected_names - actual_names)}, extra={len(actual_names - expected_names)})"
        )

print(f"Submission images: {len(pairs)}")
if missing:
    raise RuntimeError(f"Submission is incomplete; missing rendered scenes: {missing}")
if invalid_submission:
    raise RuntimeError("Submission file names/counts do not match test_poses.csv: " + "; ".join(invalid_submission))

if pairs:
    # Competition submission limit: final submission.zip should stay <= 350MB.
    # Kaggle runtime has a separate disk quota; it is not a per-file limit.
    target = 350 * 1024 * 1024
    size = pack_lossless(pairs)
    print(f"submission.zip lossless: {size / 1024 / 1024:.1f} MB")
    if size > target:
        for quality in [95, 92, 88, 85, 82, 80]:
            size = pack_as_jpeg(pairs, quality)
            print(f"submission.zip JPEG q={quality}: {size / 1024 / 1024:.1f} MB")
            if size <= target:
                break
    print(f"Saved: {SUBMISSION_ZIP}")
else:
    print("No images found; submission.zip was not created.")


# =============================================================================
# CELL 7 - Optional preview
# =============================================================================

def preview_scene(scene_name=None, n=3):
    import matplotlib.pyplot as plt
    from PIL import Image

    if scene_name is None:
        scene_dirs = sorted([p for p in SUBMISSION_DIR.iterdir() if p.is_dir()])
        if not scene_dirs:
            print("No submission scene folders to preview.")
            return
        scene_name = scene_dirs[0].name

    imgs = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        imgs.extend(sorted((SUBMISSION_DIR / scene_name).glob(ext)))
    imgs = imgs[:n]
    if not imgs:
        print(f"No images for {scene_name}")
        return

    fig, axes = plt.subplots(1, len(imgs), figsize=(6 * len(imgs), 5))
    if len(imgs) == 1:
        axes = [axes]
    for ax, img_path in zip(axes, imgs):
        ax.imshow(Image.open(img_path))
        ax.set_title(img_path.name)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


print("Notebook pipeline finished.")
