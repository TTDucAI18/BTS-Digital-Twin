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
import os
import queue
import shutil
import subprocess
import sys
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

ITERATIONS = int(os.environ.get("BTS_ITERATIONS", "40000"))
POSITION_LR_MAX_STEPS = int(os.environ.get("BTS_POSITION_LR_MAX_STEPS", str(ITERATIONS)))
if ITERATIONS <= 0 or POSITION_LR_MAX_STEPS <= 0:
    raise ValueError("BTS_ITERATIONS and BTS_POSITION_LR_MAX_STEPS must be positive.")
# Keep recovery/model-selection checkpoints at these milestones.  Checkpoint
# I/O is much cheaper than rendering a set of held-out cameras, and the latter
# is unnecessary once 40k has been selected as the submission schedule.
_requested_checkpoint_iterations = {
    int(value.strip())
    # With a 5M safety ceiling, a CUDA OOM is a hard crash rather than a clean
    # deadline stop.  Checkpoint at 10k/20k as well, while train.py retains
    # only the latest verified archive so this does not accumulate disk usage.
    for value in os.environ.get("BTS_CHECKPOINT_ITERATIONS", "10000,20000,30000,35000,40000").split(",")
    if value.strip()
}
CHECKPOINT_ITERATIONS = sorted(
    iteration for iteration in _requested_checkpoint_iterations if 0 < iteration <= ITERATIONS
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
    iteration for iteration in _requested_validation_iterations if 0 < iteration <= ITERATIONS
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
# Upper bound only, never a target.  Five million is deliberately restored for
# high-detail BTS reruns; it can require more VRAM/disk and should be paired
# with the atomic checkpoint path below.
MAX_GAUSSIANS = int(os.environ.get("BTS_MAX_GAUSSIANS", "5000000"))
# Thin tower lattice and cables retain high image-space gradients late in
# training.  Stopping at 15k freezes their allocation while the 15k--40k
# phase merely optimises oversized splats, producing the observed smearing.
DENSIFY_GRAD_THRESHOLD = float(os.environ.get("BTS_DENSIFY_GRAD_THRESHOLD", "0.00010"))
DENSIFY_UNTIL_ITER = int(os.environ.get("BTS_DENSIFY_UNTIL_ITER", "30000"))
PERCENT_DENSE = float(os.environ.get("BTS_PERCENT_DENSE", "0.005"))
if DENSIFY_GRAD_THRESHOLD <= 0 or DENSIFY_UNTIL_ITER <= 0 or not 0 < PERCENT_DENSE <= 1:
    raise ValueError("BTS densification settings must be positive, and BTS_PERCENT_DENSE must be in (0, 1].")
FOREGROUND_LOSS_WEIGHT = float(os.environ.get("BTS_FOREGROUND_LOSS_WEIGHT", "6.0"))
FOREGROUND_EDGE_LOSS_WEIGHT = float(os.environ.get("BTS_FOREGROUND_EDGE_LOSS_WEIGHT", "0.05"))
MAX_WORKERS = int(os.environ.get("BTS_MAX_WORKERS", "2"))
KAGGLE_TIME_LIMIT_H = float(os.environ.get("BTS_TIME_LIMIT_H", "11.0"))
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
# Resume an interrupted run from its own output by default.  Input checkpoints
# can be from a different data/preprocessing/configuration generation, so they
# are deliberately opt-in for quality-sensitive reruns.
RESUME_LOCAL = os.environ.get("BTS_RESUME_LOCAL", "1").strip() != "0"
RESUME_INPUT = os.environ.get("BTS_RESUME_INPUT", "0").strip() == "1"
# An explicit fresh run clears only the selected scene output/submission
# directories immediately before that scene starts.  It is opt-in because it
# intentionally discards resumable checkpoints from a prior experiment.
FRESH_RUN = os.environ.get("BTS_FRESH_RUN", "0").strip() == "1"
# Successful scenes normally release their model after their exact render set
# was copied, saving Kaggle disk.  Enable this for a rerun when final PLY and
# checkpoint artifacts must remain available for inspection or later renders.
KEEP_MODEL_ARTIFACTS = os.environ.get("BTS_KEEP_MODEL_ARTIFACTS", "0").strip() == "1"
# Offset checkpoint writes per GPU so two large archives do not saturate the
# small Kaggle working disk at the same iteration.
CHECKPOINT_STAGGER_SECONDS = float(os.environ.get("BTS_CHECKPOINT_STAGGER_SECONDS", "90"))
if (
    TRAIN_RESOLUTION <= 0
    or RENDER_RESOLUTION <= 0
    or MAX_GAUSSIANS < 0
    or FOREGROUND_LOSS_WEIGHT < 0
    or FOREGROUND_EDGE_LOSS_WEIGHT < 0
    or MAX_WORKERS <= 0
    or KAGGLE_TIME_LIMIT_H <= 0
    or KAGGLE_STOP_BUFFER_MIN < 0
    or MIN_FREE_DISK_GB < 0
    or DISK_CHECK_INTERVAL <= 0
    or WANDB_LOG_INTERVAL <= 0
    or SUBPROCESS_HEARTBEAT_SECONDS <= 0
    or not 0.0 < VALIDATION_FRACTION < 1.0
    or CHECKPOINT_STAGGER_SECONDS < 0
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
    for name in ["depths_any", "depth_anything", "depths", "depth"]:
        if (root / name).is_dir():
            return ["--depths", name]
    print(f"[{Path(scene_path).name}] Depth Anything metadata exists but no depth map directory was found; depth regularization is disabled.")
    return []


def optional_mask_args(scene_path):
    if not SUPPORTS_MASKS:
        return []
    root = train_root(scene_path)
    # Prefer the generated object masks over a generic legacy ``masks``
    # directory when both are present in a Kaggle dataset mount.
    for name in ["foreground_masks", "masks", "mask", "foreground"]:
        if (root / name).is_dir():
            args = ["--masks", name]
            if SUPPORTS_FOREGROUND_WEIGHT:
                args.extend(["--foreground_loss_weight", str(FOREGROUND_LOSS_WEIGHT)])
                args.extend(["--foreground_edge_loss_weight", str(FOREGROUND_EDGE_LOSS_WEIGHT)])
            print(f"[{Path(scene_path).name}] foreground masks: {root / name} (weight={FOREGROUND_LOSS_WEIGHT})")
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
    try:
        return int(Path(path).name.replace("chkpnt", "").replace(".pth", ""))
    except ValueError:
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
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return (
            isinstance(payload, tuple)
            and len(payload) == 2
            and isinstance(payload[1], int)
            and payload[1] == checkpoint_iter(path)
        )
    except Exception:
        return False


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
    cfg = {
        "resolution": TRAIN_RESOLUTION,
        "densify_grad_threshold": DENSIFY_GRAD_THRESHOLD,
        "densify_until_iter": min(DENSIFY_UNTIL_ITER, ITERATIONS),
        "percent_dense": PERCENT_DENSE,
        "depth_weight_init": float(os.environ.get("BTS_DEPTH_WEIGHT_INIT", "0.02")),
        "checkpoint_iterations": CHECKPOINT_ITERATIONS,
        "validation_iterations": VALIDATION_ITERATIONS,
        "max_gaussians": MAX_GAUSSIANS,
        "max_screen_size": int(os.environ.get("BTS_MAX_SCREEN_SIZE", "20")),
        # Do not erase mature thin splats during the extended densification
        # phase.  New high-gradient structure can still split until 30k.
        "opacity_reset_until_iter": int(os.environ.get("BTS_OPACITY_RESET_UNTIL_ITER", "15000")),
        # With no tower masks in the supplied scenes, this is the only loss
        # that explicitly increases gradients on cable/lattice edges.
        "image_edge_loss_weight": float(os.environ.get("BTS_IMAGE_EDGE_LOSS_WEIGHT", "0.02")),
        "position_lr_init": float(os.environ.get("BTS_POSITION_LR_INIT", "0.00016")),
        "position_lr_max_steps": POSITION_LR_MAX_STEPS,
    }
    if scene_name in CLOSEUP_SCENES:
        # bonsai/chair are compact, close-range 360-degree captures.  Smaller
        # split/clone scale and a lower gradient gate create tighter Gaussians;
        # Depth Anything remains a weak anchor instead of flattening fine shape.
        cfg.update({
            "densify_grad_threshold": float(os.environ.get("BTS_CLOSEUP_DENSIFY_GRAD_THRESHOLD", "0.00012")),
            # Do not freeze close-up geometry at 20k: this was the source of
            # the chair/bonsai point-count plateau and blurred close-up edges.
            # The final 10k remains a fixed-geometry convergence phase.
            "densify_until_iter": min(int(os.environ.get("BTS_CLOSEUP_DENSIFY_UNTIL_ITER", "30000")), ITERATIONS),
            "max_gaussians": int(os.environ.get("BTS_CLOSEUP_MAX_GAUSSIANS", "5000000")),
            "percent_dense": float(os.environ.get("BTS_CLOSEUP_PERCENT_DENSE", "0.005")),
            "depth_weight_init": float(os.environ.get("BTS_CLOSEUP_DEPTH_WEIGHT_INIT", "0.01")),
            # Keep large, distant window/background splats long enough to
            # converge, while a weak image-edge loss protects chair holes and
            # other high-contrast close-range structure without needing masks.
            "max_screen_size": int(os.environ.get("BTS_CLOSEUP_MAX_SCREEN_SIZE", "64")),
            "opacity_reset_until_iter": int(os.environ.get("BTS_CLOSEUP_OPACITY_RESET_UNTIL_ITER", "12000")),
            "image_edge_loss_weight": float(os.environ.get("BTS_CLOSEUP_IMAGE_EDGE_LOSS_WEIGHT", "0.03")),
        })
    invalid = (
        cfg["densify_grad_threshold"] <= 0
        or cfg["densify_until_iter"] <= 0
        or not 0 < cfg["percent_dense"] <= 1
        or cfg["depth_weight_init"] < 0
        or cfg["max_gaussians"] < 0
        or cfg["max_screen_size"] < 0
        or cfg["opacity_reset_until_iter"] < 0
        or cfg["image_edge_loss_weight"] < 0
        or cfg["position_lr_init"] <= 0
        or cfg["position_lr_max_steps"] <= 0
    )
    if invalid:
        raise ValueError(f"[{scene_name}] invalid scene training configuration: {cfg}")
    return cfg


def build_train_cmd(scene_path, gpu_id):
    scene_name = Path(scene_path).name
    out_dir = scene_output(scene_path)
    cfg = scene_train_config(scene_path)
    print(
        f"[{scene_name}] profile={'closeup' if scene_name in CLOSEUP_SCENES else 'bts'} "
        f"| dense={cfg['percent_dense']} | grad={cfg['densify_grad_threshold']} "
        f"| depth_weight={cfg['depth_weight_init']} | screen_prune={cfg['max_screen_size']} "
        f"| edge_loss={cfg['image_edge_loss_weight']} | checkpoints={cfg['checkpoint_iterations']} "
        f"| validate/render={cfg['validation_iterations']}"
    )
    resume = latest_checkpoint(out_dir, max_iter=ITERATIONS - 1) if (RESUME_LOCAL and not FRESH_RUN) else None
    if resume is not None:
        print(f"[{scene_name}] resuming verified local checkpoint: {resume}")
    elif RESUME_INPUT:
        resume = latest_input_checkpoint(scene_path, max_iter=ITERATIONS - 1)
    else:
        print(f"[{scene_name}] input-checkpoint resume disabled; starting clean if no local checkpoint exists.")

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
        "3",
        "--data_device",
        "cpu",
        "--iterations",
        str(ITERATIONS),
        "--lambda_dssim",
        "0.2",
        "--position_lr_init",
        str(cfg["position_lr_init"]),
        "--position_lr_max_steps",
        str(cfg["position_lr_max_steps"]),
        "--densification_interval",
        "100",
        "--densify_grad_threshold",
        str(cfg["densify_grad_threshold"]),
        "--densify_until_iter",
        str(cfg["densify_until_iter"]),
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
        str(ITERATIONS),
        "--disable_viewer",
        # Do not pass --quiet here. train.py sends it to safe_state(), whose
        # stdout wrapper drops every print and tqdm update. With subprocess
        # output streamed this otherwise makes healthy GPU workers look hung
        # while the notebook waits in as_completed().
        "--progress_name",
        f"{scene_name}-gpu{gpu_id}",
        "--skip_test_poses",
        *optional_depth_args(scene_path),
        *optional_mask_args(scene_path),
    ]
    if VALIDATION_HOLDOUT:
        cmd.append("--validation_holdout")

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
    ensemble_scales = CLOSEUP_RENDER_ENSEMBLE_SCALES if Path(scene_path).name in CLOSEUP_SCENES else RENDER_ENSEMBLE_SCALES
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
        "3",
        "--quiet",
        "--ensemble_scales",
        *[str(scale) for scale in ensemble_scales],
    ]
    if USE_ANTIALIASING:
        cmd.append("--antialiasing")
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
    out_dir = scene_output(scene_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{scene_name}] worker started on GPU {gpu_id}; running preflight.", flush=True)

    if FRESH_RUN:
        # This is intentionally scoped to exactly one selected scene.  It runs
        # before checkpoint discovery so --start_checkpoint can never revive a
        # model from the previous experiment.
        if out_dir.exists():
            shutil.rmtree(out_dir)
        scene_submission = SUBMISSION_DIR / scene_name
        if scene_submission.exists():
            shutil.rmtree(scene_submission)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{scene_name}] BTS_FRESH_RUN=1: cleared prior model and renders.")

    expected_names = expected_submission_names(scene_path)
    existing_names = submission_names(scene_name)
    if expected_names and existing_names == expected_names:
        print(f"[{scene_name}] submission already has all {len(existing_names)} expected images, skipping train/render.")
        return scene_name, 0
    if existing_names:
        print(f"[{scene_name}] incomplete/stale submission ({len(existing_names)}/{len(expected_names)}); rerendering.")

    if hours_remaining() < KAGGLE_STOP_BUFFER_MIN / 60:
        print(f"[{scene_name}] time budget exhausted, skipping.")
        return scene_name, 2

    free_gb, total_gb = disk_free_gb()
    if free_gb < MIN_FREE_DISK_GB:
        print(f"[{scene_name}] disk free {free_gb:.1f}GB < {MIN_FREE_DISK_GB:.1f}GB, skipping new training to preserve verified checkpoints.")
        return scene_name, 5

    final_ply = out_dir / "point_cloud" / f"iteration_{ITERATIONS}" / "point_cloud.ply"
    final_ckpt = out_dir / f"chkpnt{ITERATIONS}.pth"

    print(f"[{scene_name}] checking final model artifacts.", flush=True)
    if is_valid_ply(final_ply) or is_valid_checkpoint(final_ckpt):
        print(f"[{scene_name}] final model exists, skipping training.")
    else:
        print(f"[{scene_name}] preparing train.py command.", flush=True)
        cmd, env = build_train_cmd(scene_path, gpu_id)
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

        if rc != 0:
            print(f"[{scene_name}] training failed rc={rc}")
            print(tail(log, 80))
            return scene_name, rc

    # Render the requested schedule point exactly.  A prior experiment can
    # leave a later valid checkpoint in this output directory; choosing the
    # numerically latest artifact would silently submit that other experiment.
    it = ITERATIONS
    if not (is_valid_ply(final_ply) or is_valid_checkpoint(final_ckpt)):
        completed_iter = final_iteration(out_dir)
        print(
            f"[{scene_name}] training stopped at iter {completed_iter}; expected {ITERATIONS}. "
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
    if KEEP_MODEL_ARTIFACTS:
        cleanup_intermediate(out_dir, it)
        print(f"[{scene_name}] retaining final model artifacts (BTS_KEEP_MODEL_ARTIFACTS=1).")
    else:
        cleanup_intermediate(out_dir, it)
        cleanup_scene_output_after_submission(out_dir)
    free_gb, total_gb = disk_free_gb()
    print(f"[{scene_name}] done. Disk: {free_gb:.1f}/{total_gb:.1f} GB free")
    return scene_name, 0


# =============================================================================
# CELL 5 - Run two-GPU queue
# =============================================================================

def scene_priority(scene_path):
    out_dir = scene_output(scene_path)
    final_ply = out_dir / "point_cloud" / f"iteration_{ITERATIONS}" / "point_cloud.ply"
    if is_valid_ply(final_ply) or is_valid_checkpoint(out_dir / f"chkpnt{ITERATIONS}.pth"):
        return (3, 0)
    partial = latest_checkpoint(out_dir, max_iter=ITERATIONS - 1) if RESUME_LOCAL else None
    if partial:
        # Finish the most advanced resumable models first.
        return (2, checkpoint_iter(partial))
    # BTS is the submission priority.  Starting bonsai/chair on both GPUs
    # previously pushed all five tower scenes into the global deadline.
    return (1 if Path(scene_path).name not in CLOSEUP_SCENES else 0, 0)


ALL_SCENES = sorted(ALL_SCENES, key=scene_priority, reverse=True)
gpu_queue = queue.Queue()
for gpu in GPU_IDS:
    gpu_queue.put(gpu)


def worker(scene):
    gpu = gpu_queue.get()
    try:
        print(f"[{Path(scene).name}] acquired GPU {gpu} from queue.", flush=True)
        return train_and_render_scene(scene, gpu)
    finally:
        gpu_queue.put(gpu)


print("=" * 80)
print(
    f"Starting pipeline: {len(ALL_SCENES)} scenes, GPUs={GPU_IDS}, iterations={ITERATIONS}, "
    f"checkpoints={CHECKPOINT_ITERATIONS}, validation/render={VALIDATION_ITERATIONS}"
)
print("=" * 80)

_scene_start_times: dict = {}

results = []
_pipeline_bar = _tqdm(total=len(ALL_SCENES), desc="Scenes", unit="scene", dynamic_ncols=True)
executor = ThreadPoolExecutor(max_workers=len(GPU_IDS))
futures = {}
try:
    futures = {executor.submit(worker, scene): scene for scene in ALL_SCENES}
    for future in as_completed(futures):
        scene = futures[future]
        scene_name = Path(scene).name
        t_done = time.time()
        try:
            result = future.result()
        except Exception as exc:
            result = (scene_name, 99)
            print(f"[{scene_name}] unhandled error: {exc}")
        results.append(result)
        rc = result[1]
        rc_label = {0: "OK", 2: "timeout", 3: "no-PLY", 4: "no-renders",
                    5: "low-disk", 6: "partial", 7: "invalid-render-set", 99: "exception"}.get(rc, f"rc={rc}")
        free_gb, _ = disk_free_gb()
        n_imgs = submission_image_count(scene_name)
        _pipeline_bar.set_postfix(scene=scene_name, status=rc_label, imgs=n_imgs,
                                  disk_free=f"{free_gb:.1f}GB")
        _pipeline_bar.update(1)
        print(f"Completed: {result} | imgs={n_imgs} | disk_free={free_gb:.1f}GB")
        # WandB belongs to child training processes.  The notebook parent has
        # no wandb.init(), so do not emit a false error after pipeline exit.
except KeyboardInterrupt:
    # Do not let ThreadPoolExecutor.__exit__ wait indefinitely for subprocesses
    # after an interrupted Kaggle cell.  Stop active children first, then join
    # the short-lived reader/worker threads.
    print("KeyboardInterrupt: stopping active train/render subprocesses...")
    for future in futures:
        future.cancel()
    stop_active_processes()
    # A worker can be blocked in a slow checkpoint/PLY validation before it
    # has spawned a child process.  Do not make the notebook wait for that
    # thread after the user has explicitly interrupted the cell.
    executor.shutdown(wait=False, cancel_futures=True)
    raise
except BaseException:
    stop_active_processes()
    executor.shutdown(wait=False, cancel_futures=True)
    raise
else:
    executor.shutdown(wait=True)
finally:
    _pipeline_bar.close()

print("Pipeline results:", results)

# WandB: log final pipeline summary table
if USE_WANDB:
    try:
        import wandb as _wandb
        _rows = [[name, rc, {0:"OK",2:"timeout",3:"no-PLY",4:"no-renders",
                              5:"low-disk",6:"partial",7:"invalid-render-set",99:"exception"}.get(rc, str(rc)),
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
