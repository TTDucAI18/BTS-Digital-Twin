# =============================================================================
# BTS Digital Twin - Kaggle notebook script
#
# Copy this file into Kaggle as code cells, or run it as one Python script.
# The pipeline is tuned for Kaggle 2x T4, 16 GB VRAM per GPU:
#   1. clone/update repo and submodules
#   2. install CUDA extensions once
#   3. discover the 8 private_set1 scenes
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
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# =============================================================================
# CELL 1 - Configuration
# =============================================================================

REPO_URL = os.environ.get("BTS_REPO_URL", "https://github.com/TTDucAI18/BTS-Digital-Twin.git")
REPO_DIR = Path(os.environ.get("BTS_REPO_DIR", "/kaggle/working/BTS-Digital-Twin"))
OUTPUT_DIR = Path(os.environ.get("BTS_OUTPUT_DIR", "/kaggle/working/output"))
SUBMISSION_DIR = Path(os.environ.get("BTS_SUBMISSION_DIR", "/kaggle/working/submission"))
SUBMISSION_ZIP = Path(os.environ.get("BTS_SUBMISSION_ZIP", "/kaggle/working/submission.zip"))

DATA_ROOT_CANDIDATES = [
    Path(os.environ["BTS_DATA_DIR"]) if os.environ.get("BTS_DATA_DIR") else None,
    Path("/kaggle/input/datasets/tdukaggle/ai-race-data/phase1"),
    Path("/kaggle/input/bts-digital-twin-phase1/phase1"),
    Path("/kaggle/input/ai-race-data/phase1"),
    Path("D:/ai_race_2026/data/phase1"),
]

ITERATIONS = int(os.environ.get("BTS_ITERATIONS", "30000"))
TRAIN_RESOLUTION = int(os.environ.get("BTS_TRAIN_RESOLUTION", "2"))
RENDER_RESOLUTION = int(os.environ.get("BTS_RENDER_RESOLUTION", "1"))
MAX_GAUSSIANS = int(os.environ.get("BTS_MAX_GAUSSIANS", "2500000"))
MAX_WORKERS = int(os.environ.get("BTS_MAX_WORKERS", "2"))
KAGGLE_TIME_LIMIT_H = float(os.environ.get("BTS_TIME_LIMIT_H", "11.0"))
PRIVATE_SET1_SCENES = [
    "HCM0249",
    "HCM0254",
    "HCM0276",
    "HCM1439",
    "HNI0131",
    "HNI0265",
    "HNI0366",
    "HNI0437",
]
SCENE_FILTER = os.environ.get("BTS_SCENES", ",".join(PRIVATE_SET1_SCENES)).strip()

WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "").strip()
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "").strip()
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "bts-digital-twin")
USE_WANDB = bool(WANDB_API_KEY)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

SESSION_START = time.time()


def run(cmd, cwd=None, log_file=None, check=False, env=None):
    """Run a command and stream or capture output."""
    cwd = str(cwd) if cwd is not None else None
    printable = " ".join(str(x) for x in cmd)
    print(f"\n$ {printable}")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8", errors="replace") as f:
            result = subprocess.run(
                [str(x) for x in cmd],
                cwd=cwd,
                env=merged_env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
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

    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with rc={result.returncode}: {printable}")
    return result.returncode


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

if REPO_DIR.exists():
    run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=False)
    run(["git", "-C", REPO_DIR, "submodule", "update", "--init", "--recursive"], check=True)
else:
    run(["git", "clone", "--recurse-submodules", REPO_URL, REPO_DIR], check=True)

os.chdir(REPO_DIR)
run(["git", "log", "--oneline", "-3"], cwd=REPO_DIR, check=False)


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

run([sys.executable, "-m", "pip", "install", "-q", "plyfile", "tqdm", "opencv-python-headless", "ninja", "Pillow", "matplotlib", "setuptools<70.0.0"], check=True)
if USE_WANDB:
    run([sys.executable, "-m", "pip", "install", "-q", "wandb"], check=True)

for submodule in [
    REPO_DIR / "submodules" / "diff-gaussian-rasterization",
    REPO_DIR / "submodules" / "simple-knn",
    REPO_DIR / "submodules" / "fused-ssim",
]:
    if submodule.exists():
        run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", submodule], cwd=REPO_DIR, check=True)
    else:
        print(f"WARNING: missing submodule {submodule}")

verify_code = (
    "from diff_gaussian_rasterization import GaussianRasterizer; "
    "from simple_knn._C import distCUDA2; "
    "print('CUDA extensions OK')"
)
run([sys.executable, "-c", verify_code], cwd=REPO_DIR, check=True)

if USE_WANDB:
    import wandb

    wandb.login(key=WANDB_API_KEY)
    if not WANDB_ENTITY:
        try:
            WANDB_ENTITY = wandb.Api().viewer.username
        except Exception:
            WANDB_ENTITY = ""
    print(f"WandB enabled: project={WANDB_PROJECT}, entity={WANDB_ENTITY or '<default>'}")
else:
    print("WandB disabled. Set WANDB_API_KEY in Kaggle Secrets to enable it.")

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
            if (candidate / "private_set1").exists():
                return candidate
    raise FileNotFoundError(
        "Could not find phase1 data root. Set BTS_DATA_DIR to the folder containing private_set1."
    )


def is_scene_dir(path):
    path = Path(path)
    return (path / "train" / "sparse").exists() or (path / "sparse").exists()


def discover_scenes(data_root):
    split_dir = data_root / "private_set1"
    if not split_dir.exists():
        raise FileNotFoundError(f"private_set1 not found under {data_root}")

    if SCENE_FILTER:
        wanted = {x.strip() for x in SCENE_FILTER.split(",") if x.strip()}
    else:
        wanted = set(PRIVATE_SET1_SCENES)

    unknown = sorted(wanted - set(PRIVATE_SET1_SCENES))
    if unknown:
        raise ValueError(f"BTS_SCENES contains names outside private_set1 target list: {unknown}")

    scenes = [split_dir / name for name in PRIVATE_SET1_SCENES if name in wanted]
    missing_dirs = [p.name for p in scenes if not p.exists()]
    invalid_dirs = [p.name for p in scenes if p.exists() and not is_scene_dir(p)]
    if missing_dirs:
        raise FileNotFoundError(f"Missing private_set1 scene dirs: {missing_dirs}")
    if invalid_dirs:
        raise RuntimeError(f"Invalid private_set1 scene dirs, missing train/sparse or sparse: {invalid_dirs}")

    if not scenes:
        raise RuntimeError(f"No selected private_set1 scenes found under {split_dir}")
    return scenes


DATA_ROOT = find_data_root()
ALL_SCENES = discover_scenes(DATA_ROOT)
print(f"DATA_ROOT: {DATA_ROOT}")
print(f"Private set1 scenes ({len(ALL_SCENES)}): {[p.name for p in ALL_SCENES]}")


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
        return []
    for name in ["depths_any", "depth_anything", "depths", "depth"]:
        if (root / name).is_dir():
            return ["--depths", name]
    return []


def optional_mask_args(scene_path):
    if not SUPPORTS_MASKS:
        return []
    root = train_root(scene_path)
    for name in ["masks", "mask", "foreground_masks", "foreground"]:
        if (root / name).is_dir():
            args = ["--masks", name]
            if SUPPORTS_FOREGROUND_WEIGHT:
                args.extend(["--foreground_loss_weight", "4.0"])
            return args
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
            return zf.testzip() is None
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


def final_iteration(out_dir):
    pc_dir = Path(out_dir) / "point_cloud"
    iters = []
    if pc_dir.exists():
        for p in pc_dir.glob("iteration_*"):
            try:
                if (p / "point_cloud.ply").exists():
                    iters.append(int(p.name.replace("iteration_", "")))
            except ValueError:
                pass
    ckpt = latest_checkpoint(out_dir)
    if ckpt:
        iters.append(checkpoint_iter(ckpt))
    return max(iters) if iters else ITERATIONS


def ensure_ply_from_checkpoint(out_dir, iteration):
    out_dir = Path(out_dir)
    ply = out_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if ply.exists():
        return True

    ckpt = out_dir / f"chkpnt{iteration}.pth"
    if not is_valid_checkpoint(ckpt):
        return False

    helper = Path("/kaggle/working/extract_checkpoint_ply.py")
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
    return rc == 0 and ply.exists()


def scene_train_config(scene_path):
    n = count_images(scene_path)
    cfg = {
        "resolution": TRAIN_RESOLUTION,
        "densify_grad_threshold": 0.00020,
        "densify_until_iter": min(12000, ITERATIONS),
        "checkpoint_iterations": sorted({7500, 15000, 25000, ITERATIONS}),
        "max_gaussians": MAX_GAUSSIANS,
    }
    if n <= 180:
        cfg["densify_grad_threshold"] = 0.00016
        cfg["densify_until_iter"] = min(15000, ITERATIONS)
    elif n >= 280:
        cfg["densify_grad_threshold"] = 0.00024
        cfg["densify_until_iter"] = min(10000, ITERATIONS)
    return cfg


def build_train_cmd(scene_path, gpu_id):
    out_dir = scene_output(scene_path)
    cfg = scene_train_config(scene_path)
    resume = latest_checkpoint(out_dir, max_iter=ITERATIONS - 1)

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
        "0.00016",
        "--densification_interval",
        "100",
        "--densify_grad_threshold",
        str(cfg["densify_grad_threshold"]),
        "--densify_until_iter",
        str(cfg["densify_until_iter"]),
        "--opacity_reset_interval",
        "3000",
        "--depth_weight_init",
        "0.02",
        "--checkpoint_iterations",
        *[str(x) for x in cfg["checkpoint_iterations"]],
        "--save_iterations",
        str(ITERATIONS),
        "--disable_viewer",
        "--quiet",
        *optional_depth_args(scene_path),
        *optional_mask_args(scene_path),
    ]

    if SUPPORTS_MAX_GAUSSIANS:
        cmd.extend(["--max_gaussians", str(cfg["max_gaussians"])])

    if resume:
        cmd.extend(["--start_checkpoint", resume])
    if USE_WANDB:
        cmd.extend(["--use_wandb", "--wandb_project", WANDB_PROJECT])
        if WANDB_ENTITY:
            cmd.extend(["--wandb_entity", WANDB_ENTITY])

    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    return cmd, env


def build_render_cmd(scene_path, gpu_id, iteration):
    out_dir = scene_output(scene_path)
    return [
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
    ], {"CUDA_VISIBLE_DEVICES": str(gpu_id)}


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


def copy_renders_to_submission(scene_path, iteration):
    scene_name = Path(scene_path).name
    out_dir = scene_output(scene_path)
    render_dir = out_dir / "test" / f"ours_{iteration}" / "renders"
    dest = SUBMISSION_DIR / scene_name
    if not render_dir.exists():
        print(f"[{scene_name}] missing render dir: {render_dir}")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    images = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]:
        images.extend(sorted(render_dir.glob(ext)))

    for img in images:
        shutil.copy2(img, dest / img.name)
    print(f"[{scene_name}] copied {len(images)} renders to {dest}")
    return len(images)


def train_and_render_scene(scene_path, gpu_id):
    scene_path = Path(scene_path)
    scene_name = scene_path.name
    out_dir = scene_output(scene_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if hours_remaining() < 0.25:
        print(f"[{scene_name}] time budget exhausted, skipping.")
        return scene_name, 2

    final_ply = out_dir / "point_cloud" / f"iteration_{ITERATIONS}" / "point_cloud.ply"
    final_ckpt = out_dir / f"chkpnt{ITERATIONS}.pth"

    if final_ply.exists() or is_valid_checkpoint(final_ckpt):
        print(f"[{scene_name}] final model exists, skipping training.")
    else:
        cmd, env = build_train_cmd(scene_path, gpu_id)
        log = OUTPUT_DIR / f"{scene_name}_train.log"
        print(f"[{scene_name}] train on GPU {gpu_id} | images={count_images(scene_path)} | log={log}")
        rc = run(cmd, cwd=REPO_DIR, env=env, log_file=log, check=False)
        if rc != 0:
            print(f"[{scene_name}] training failed rc={rc}")
            print(tail(log, 80))
            return scene_name, rc

    it = final_iteration(out_dir)
    if not ensure_ply_from_checkpoint(out_dir, it):
        ply = out_dir / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
        print(f"[{scene_name}] no PLY found after training: {ply}")
        return scene_name, 3

    cmd, env = build_render_cmd(scene_path, gpu_id, it)
    log = OUTPUT_DIR / f"{scene_name}_render.log"
    print(f"[{scene_name}] render iteration {it} on GPU {gpu_id} | log={log}")
    rc = run(cmd, cwd=REPO_DIR, env=env, log_file=log, check=False)
    if rc != 0:
        print(f"[{scene_name}] render failed rc={rc}")
        print(tail(log, 80))
        return scene_name, rc

    n = copy_renders_to_submission(scene_path, it)
    if n > 0:
        cleanup_intermediate(out_dir, it)
        free_gb, total_gb = disk_free_gb()
        print(f"[{scene_name}] done. Disk: {free_gb:.1f}/{total_gb:.1f} GB free")
        return scene_name, 0

    return scene_name, 4


# =============================================================================
# CELL 5 - Run two-GPU queue
# =============================================================================

def scene_priority(scene_path):
    out_dir = scene_output(scene_path)
    final_ply = out_dir / "point_cloud" / f"iteration_{ITERATIONS}" / "point_cloud.ply"
    if final_ply.exists() or is_valid_checkpoint(out_dir / f"chkpnt{ITERATIONS}.pth"):
        return 0
    if latest_checkpoint(out_dir, max_iter=ITERATIONS - 1):
        return 2
    return 1


ALL_SCENES = sorted(ALL_SCENES, key=scene_priority, reverse=True)
gpu_queue = queue.Queue()
for gpu in GPU_IDS:
    gpu_queue.put(gpu)


def worker(scene):
    gpu = gpu_queue.get()
    try:
        return train_and_render_scene(scene, gpu)
    finally:
        gpu_queue.put(gpu)


print("=" * 80)
print(f"Starting pipeline: {len(ALL_SCENES)} scenes, GPUs={GPU_IDS}, iterations={ITERATIONS}")
print("=" * 80)

results = []
with ThreadPoolExecutor(max_workers=len(GPU_IDS)) as executor:
    futures = {executor.submit(worker, scene): scene for scene in ALL_SCENES}
    for future in as_completed(futures):
        scene = futures[future]
        try:
            result = future.result()
        except Exception as exc:
            result = (Path(scene).name, 99)
            print(f"[{Path(scene).name}] unhandled error: {exc}")
        results.append(result)
        print(f"Completed: {result}")

print("Pipeline results:", results)


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
        for full, arcname in pairs:
            zf.write(full, arcname)
    return SUBMISSION_ZIP.stat().st_size


def pack_as_jpeg(pairs, quality):
    from PIL import Image

    if SUBMISSION_ZIP.exists():
        SUBMISSION_ZIP.unlink()
    with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for full, arcname in pairs:
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
for scene in ALL_SCENES:
    scene_dir = SUBMISSION_DIR / scene.name
    if not scene_dir.exists() or not any(scene_dir.glob("*")):
        missing.append(scene.name)

print(f"Submission images: {len(pairs)}")
if missing:
    print(f"WARNING: missing rendered scenes: {missing}")

if pairs:
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
