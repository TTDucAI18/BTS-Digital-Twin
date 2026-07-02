# =============================================================================
# BTS Digital Twin — Kaggle Notebook
# Copy từng cell vào Kaggle Notebook (Code cell)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Kiểm tra môi trường
# ─────────────────────────────────────────────────────────────────────────────
import os, subprocess

def run(cmd, **kwargs):
    """Helper: chạy shell command và print output realtime."""
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True, **kwargs)
    if result.stdout: print(result.stdout)
    if result.stderr: print(result.stderr)
    return result.returncode

print("=" * 60)
print("CUDA & PyTorch versions")
print("=" * 60)
run("nvcc --version")
run("python -c \"import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda}')\"")
run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — (Đã xoá: Không cài lại PyTorch để tránh lỗi mismatch với Kaggle NVCC 12.x)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Sử dụng PyTorch mặc định của Kaggle để đảm bảo khớp với NVCC...")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Clone repo và cài dependencies
# ─────────────────────────────────────────────────────────────────────────────
REPO_URL  = "https://github.com/TTDucAI18/BTS-Digital-Twin.git"
REPO_DIR  = "/kaggle/working/BTS-Digital-Twin"

print("=" * 60)
print("Cloning BTS-Digital-Twin repo...")
print("=" * 60)

if os.path.exists(REPO_DIR):
    print(f"Repo already exists at {REPO_DIR}, pulling latest...")
    run(f"git -C {REPO_DIR} pull origin main")
    run(f"git -C {REPO_DIR} submodule update --init --recursive")
else:
    run(f"git clone --recurse-submodules {REPO_URL} {REPO_DIR}")

os.chdir(REPO_DIR)
print(f"\nWorking directory: {os.getcwd()}")
run("git log --oneline -3")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Cài đặt submodules (diff-gaussian-rasterization, simple-knn, fused-ssim)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Installing Python dependencies & submodules...")
print("=" * 60)

# Base deps (Bắt buộc dùng setuptools < 70 để tránh lỗi distutils khi compile CUDA)
run("pip install plyfile tqdm wandb opencv-python ninja \"setuptools<70.0.0\"")

# Compile submodules
for submod in [
    "submodules/diff-gaussian-rasterization",
    "submodules/simple-knn",
    "submodules/fused-ssim",
]:
    print(f"\n[Building] {submod} ...")
    rc = run(f"pip install --no-build-isolation -e {submod}", cwd=REPO_DIR)
    status = "✅ OK" if rc == 0 else "❌ FAILED"
    print(f"  → {status}")

# Verify import
run("python -c \"from diff_gaussian_rasterization import GaussianRasterizer; print('Rasterizer OK')\"", cwd=REPO_DIR)
run("python -c \"import simple_knn; print('simple_knn OK')\"", cwd=REPO_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — WandB login
# ─────────────────────────────────────────────────────────────────────────────
import wandb

print("=" * 60)
print("Logging in to Weights & Biases...")
print("=" * 60)
wandb.login(key="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")
print("WandB login OK ✅")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Cấu hình đường dẫn data & output
# ─────────────────────────────────────────────────────────────────────────────
import glob

DATA_DIR   = "/kaggle/input/datasets/tdukaggle/ai-race-data/phase1"
OUTPUT_DIR = "/kaggle/working/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thu thập tất cả scenes (bỏ qua file .DS_Store và __MACOSX)
public_scenes  = sorted([
    p for p in glob.glob(f"{DATA_DIR}/public_set/*")
    if os.path.isdir(p) and not os.path.basename(p).startswith(".")
])
private_scenes = sorted([
    p for p in glob.glob(f"{DATA_DIR}/private_set1/*")
    if os.path.isdir(p) and not os.path.basename(p).startswith(".")
])
all_scenes = public_scenes + private_scenes

print(f"Public  scenes ({len(public_scenes)}): {[os.path.basename(s) for s in public_scenes]}")
print(f"Private scenes ({len(private_scenes)}): {[os.path.basename(s) for s in private_scenes]}")
print(f"Total: {len(all_scenes)} scenes")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Chạy Training Multi-GPU (scene chẵn → GPU 0, scene lẻ → GPU 1)
# ─────────────────────────────────────────────────────────────────────────────
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_DIR   = "/kaggle/working/BTS-Digital-Twin"
OUTPUT_DIR = "/kaggle/working/output"
ITERATIONS = 30000

def train_scene(scene_path: str, gpu_id: int) -> str:
    scene_name = os.path.basename(scene_path)
    scene_out  = f"{OUTPUT_DIR}/{scene_name}"
    log_file   = f"{OUTPUT_DIR}/{scene_name}_train.log"
    os.makedirs(scene_out, exist_ok=True)

    # Auto-resume: tìm checkpoint mới nhất
    ckpts = sorted(glob.glob(f"{scene_out}/chkpnt*.pth"))
    resume_flag = f"--start_checkpoint {ckpts[-1]}" if ckpts else ""
    if resume_flag:
        print(f"  [{scene_name}] Resuming from {os.path.basename(ckpts[-1])}")

    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} python {REPO_DIR}/train.py "
        f"-s {scene_path} "
        f"-m {scene_out} "
        f"--use_wandb "
        f"--wandb_project bts-digital-twin-kaggle "
        f"--iterations {ITERATIONS} "
        f"--lambda_dssim 0.4 "
        f"--densify_grad_threshold 0.00015 "
        f"--checkpoint_iterations 7000 15000 30000 "
        f"--disable_viewer "
        f"{resume_flag}"
    )

    print(f"\n🚀 Training [{scene_name}] on GPU {gpu_id} ...")
    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT, cwd=REPO_DIR)

    status = "✅ DONE" if result.returncode == 0 else f"❌ FAILED (rc={result.returncode})"
    print(f"  [{scene_name}] {status} — log: {log_file}")
    
    if result.returncode != 0:
        print(f"\n{'='*20} ERROR LOG FOR {scene_name} (Last 50 lines) {'='*20}")
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                print("".join(lines[-50:]))
        except Exception as e:
            print(f"Could not read log: {e}")
        print("=" * 60 + "\n")
        
    return scene_name, result.returncode


# Chạy song song: mỗi cặp 2 scenes trên 2 GPU
print("=" * 60)
print(f"Starting training for {len(all_scenes)} scenes on 2x GPU T4...")
print("=" * 60)

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        executor.submit(train_scene, scene, i % 2): scene
        for i, scene in enumerate(all_scenes)
    }
    for future in as_completed(futures):
        scene_name, rc = future.result()

print("\nAll training jobs completed!")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Render test poses (sinh ảnh submission)
# ─────────────────────────────────────────────────────────────────────────────
def render_scene(scene_path: str, gpu_id: int) -> str:
    scene_name = os.path.basename(scene_path)
    scene_out  = f"{OUTPUT_DIR}/{scene_name}"
    log_file   = f"{OUTPUT_DIR}/{scene_name}_render.log"

    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} python {REPO_DIR}/render.py "
        f"-m {scene_out} "
        f"--skip_train "
        f"--iteration {ITERATIONS}"
    )

    print(f"🎨 Rendering [{scene_name}] on GPU {gpu_id} ...")
    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT, cwd=REPO_DIR)

    status = "✅ DONE" if result.returncode == 0 else f"❌ FAILED (rc={result.returncode})"
    print(f"  [{scene_name}] {status}")
    
    if result.returncode != 0:
        print(f"\n{'='*20} ERROR LOG FOR {scene_name} (Last 50 lines) {'='*20}")
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                print("".join(lines[-50:]))
        except Exception as e:
            print(f"Could not read log: {e}")
        print("=" * 60 + "\n")

    return scene_name, result.returncode


print("=" * 60)
print("Rendering test views for all scenes...")
print("=" * 60)

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        executor.submit(render_scene, scene, i % 2): scene
        for i, scene in enumerate(all_scenes)
    }
    for future in as_completed(futures):
        scene_name, rc = future.result()

print("\nAll renders completed!")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Đóng gói submission.zip
# ─────────────────────────────────────────────────────────────────────────────
import zipfile, shutil

SUBMISSION_DIR = "/kaggle/working/submission"
SUBMISSION_ZIP = "/kaggle/working/submission.zip"

print("=" * 60)
print("Packaging submission.zip ...")
print("=" * 60)

os.makedirs(SUBMISSION_DIR, exist_ok=True)
missing_scenes = []

for scene_path in all_scenes:
    scene_name  = os.path.basename(scene_path)
    render_path = f"{OUTPUT_DIR}/{scene_name}/test/ours_{ITERATIONS}/renders"

    if not os.path.isdir(render_path):
        print(f"  ⚠️  [{scene_name}] Render path not found: {render_path}")
        missing_scenes.append(scene_name)
        continue

    dest = f"{SUBMISSION_DIR}/{scene_name}"
    os.makedirs(dest, exist_ok=True)

    imgs = glob.glob(f"{render_path}/*.png")
    for img in imgs:
        shutil.copy(img, dest)

    print(f"  ✅ [{scene_name}] Copied {len(imgs)} images")

# Zip
if os.path.exists(SUBMISSION_ZIP):
    os.remove(SUBMISSION_ZIP)

with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(SUBMISSION_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            arcname   = os.path.relpath(full_path, SUBMISSION_DIR)
            zf.write(full_path, arcname)

zip_size_mb = os.path.getsize(SUBMISSION_ZIP) / 1024 / 1024
print(f"\n✅ submission.zip created: {SUBMISSION_ZIP} ({zip_size_mb:.1f} MB)")

if missing_scenes:
    print(f"\n⚠️  Missing renders for: {missing_scenes}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 (tuỳ chọn) — Xem preview một số ảnh render
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
from PIL import Image

def preview_scene(scene_name: str, n: int = 3):
    render_path = f"{OUTPUT_DIR}/{scene_name}/test/ours_{ITERATIONS}/renders"
    imgs = sorted(glob.glob(f"{render_path}/*.png"))[:n]
    if not imgs:
        print(f"No renders found for {scene_name}")
        return
    fig, axes = plt.subplots(1, len(imgs), figsize=(6 * len(imgs), 5))
    if len(imgs) == 1: axes = [axes]
    for ax, img_path in zip(axes, imgs):
        ax.imshow(Image.open(img_path))
        ax.set_title(os.path.basename(img_path))
        ax.axis("off")
    plt.suptitle(f"Scene: {scene_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

# Preview scene đầu tiên
if all_scenes:
    preview_scene(os.path.basename(all_scenes[0]))
