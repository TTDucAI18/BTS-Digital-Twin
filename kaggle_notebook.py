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
# CELL 2 — Sử dụng PyTorch mặc định của Kaggle (tránh lỗi mismatch với NVCC 12.x)
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

# Base deps — setuptools<70 tránh lỗi distutils khi compile CUDA extension
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

# Thu thập tất cả scenes (bỏ qua .DS_Store và __MACOSX)
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
# CELL 7 — Training Multi-GPU (queue-based: GPU nào rảnh nhận scene tiếp theo)
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, queue, shutil, time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_DIR             = "/kaggle/working/BTS-Digital-Twin"
OUTPUT_DIR           = "/kaggle/working/output"
ITERATIONS           = 30_000   # Train đầy đủ 30k iters
FINETUNE_ITERS       = 0        # Fine-tune TẮT (Phase 2 gây regression trên BTS data)
KAGGLE_TIME_LIMIT_H  = 10.0     # Ngưỡng an toàn (12h limit Kaggle − 2h buffer)
KAGGLE_SESSION_START = time.time()

# Thư mục chứa checkpoint từ Kaggle Input dataset (để resume nếu có)
INPUT_CHECKPOINT_DIR = "/kaggle/input/datasets/tdukaggle/ai-race-data"


def check_disk_space(label: str = ""):
    """In dung lượng còn trống trên /kaggle/working."""
    total, used, free = shutil.disk_usage("/kaggle/working")
    free_gb  = free  / (1024 ** 3)
    total_gb = total / (1024 ** 3)
    tag  = f"[{label}] " if label else ""
    flag = "⚠️ LOW DISK" if free_gb < 3 else "💾"
    print(f"  {flag} {tag}Disk: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
    return free_gb


def cleanup_after_train(scene_out: str, scene_name: str, final_iter: int = None):
    """Giải phóng disk sau khi train xong một scene.
    - Xoá thư mục point_cloud của iteration cũ (chỉ giữ mới nhất)
    - Xoá tensorboard event files
    - KHÔNG xoá checkpoint — đã được xử lý trong train_scene() sau khi verify.
    """
    if final_iter is None:
        final_iter = get_final_iteration(scene_out)
    freed = 0

    # Xóa Tensorboard event files (nếu vô tình sinh ra)
    for tfevent in glob.glob(f"{scene_out}/events.out.tfevents*"):
        try:
            size = os.path.getsize(tfevent)
            os.remove(tfevent)
            freed += size
        except: pass

    # Xoá wandb offline run directories để nhẹ máy
    wandb_dir = "/kaggle/working/wandb"
    if os.path.exists(wandb_dir):
        for run_dir in os.listdir(wandb_dir):
            if run_dir.startswith("run-") or run_dir.startswith("offline-"):
                try:
                    shutil.rmtree(os.path.join(wandb_dir, run_dir))
                except: pass

    # Xoá point_cloud của iteration cũ, chỉ giữ iteration mới nhất
    pc_base   = f"{scene_out}/point_cloud"
    keep_iter = f"iteration_{final_iter}"
    if os.path.isdir(pc_base):
        for iter_dir in os.listdir(pc_base):
            if iter_dir != keep_iter:
                full = os.path.join(pc_base, iter_dir)
                try:
                    size = sum(os.path.getsize(os.path.join(dp, f))
                               for dp, _, fs in os.walk(full) for f in fs)
                    shutil.rmtree(full)
                    freed += size
                    print(f"    🗑  Removed old PLY dir: point_cloud/{iter_dir}")
                except Exception as e:
                    print(f"    ⚠️  Could not remove {full}: {e}")

    if freed > 0:
        print(f"  [{scene_name}] 🧹 Cleanup freed: {freed / (1024**3):.2f} GB")
    check_disk_space(scene_name)


def is_valid_checkpoint(path: str) -> bool:
    """Kiểm tra checkpoint có phải ZIP hợp lệ không (PyTorch .pth là zip archive).
    File bị truncate do disk-full sẽ thiếu magic bytes và central directory.
    """
    import zipfile
    try:
        with zipfile.ZipFile(path, 'r') as zf:
            bad = zf.testzip()   # None nếu tất cả OK
            return bad is None
    except (zipfile.BadZipFile, EOFError, OSError):
        return False


def has_time_remaining(hours_needed: float) -> bool:
    """Kiểm tra session còn đủ thời gian để chạy thêm bước không."""
    elapsed_h   = (time.time() - KAGGLE_SESSION_START) / 3600
    remaining_h = KAGGLE_TIME_LIMIT_H - elapsed_h
    if remaining_h < hours_needed:
        print(f"  ⏱ Thời gian còn lại: {remaining_h:.1f}h < cần {hours_needed:.1f}h → bỏ qua bước này.")
        return False
    print(f"  ⏱ Thời gian còn lại: {remaining_h:.1f}h → đủ để chạy thêm {hours_needed:.1f}h")
    return True


def get_final_iteration(scene_out: str) -> int:
    """Tìm iteration cao nhất có sẵn trong thư mục point_cloud."""
    pc_base = os.path.join(scene_out, "point_cloud")
    if not os.path.isdir(pc_base):
        return ITERATIONS
    iters = []
    for d in os.listdir(pc_base):
        if d.startswith("iteration_"):
            try:
                iters.append(int(d.replace("iteration_", "")))
            except ValueError:
                pass
    return max(iters) if iters else ITERATIONS


def get_scene_config(scene_path: str) -> dict:
    """Trả về config training tối ưu dựa trên số ảnh của scene.

    Chiến lược -r 2:
    - -r 2 → ảnh ~660×494, VRAM giảm ~4x, tốc độ tăng ~2-3x so với -r 1.
    - Bù bằng: densify_grad_threshold thấp hơn, densify_until_iter cao hơn (22000).
    """
    # BTS structure: scene_path/train/images
    img_dir = os.path.join(scene_path, "train", "images")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(scene_path, "images")   # fallback
    if os.path.isdir(img_dir):
        n_imgs = len([f for f in os.listdir(img_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    else:
        n_imgs = 240   # Không detect được → assume trung bình
    print(f"  [Config] Số ảnh train: {n_imgs}")

    if n_imgs <= 120:
        # Scene nhỏ: ngưỡng nhạy hơn một chút để bắt chi tiết
        return {"resolution": 2, "densify_until_iter": 15000, "densify_grad_threshold": 0.00015}
    elif n_imgs <= 220:
        # Scene vừa
        return {"resolution": 2, "densify_until_iter": 15000, "densify_grad_threshold": 0.0002}
    else:
        # Scene đầy đủ: ngưỡng chuẩn của 3DGS
        return {"resolution": 2, "densify_until_iter": 15000, "densify_grad_threshold": 0.00025}


def finetune_scene(scene_path: str, scene_out: str, gpu_id: int) -> int:
    """Phase 2: Fine-tune tại -r 1 (full resolution) sau khi Phase 1 xong.

    Tại sao an toàn về VRAM:
    - densify_until_iter=0 → không chạy densification → VRAM tiết kiệm đáng kể.
    - Gaussians đã ở đúng vị trí từ Phase 1; Phase 2 chỉ tinh chỉnh
      màu sắc/opacity tại full-res → gains về PSNR/SSIM.
    - Exposure compensation BỊ TẮT (giống Phase 1) vì BTS data đồng đều ánh sáng.
    """
    scene_name = os.path.basename(scene_path)
    if FINETUNE_ITERS <= 0:
        return 0

    # Ước tính ~8-12 phút/scene cho 5000 iters → cần ít nhất 0.25h buffer
    if not has_time_remaining(0.25):
        print(f"  [{scene_name}] Phase 2 bị bỏ qua do sắp hết thời gian.")
        return 0

    # Tìm checkpoint Phase 1
    p1_ckpt = f"{scene_out}/chkpnt{ITERATIONS}.pth"
    if not is_valid_checkpoint(p1_ckpt):
        print(f"  [{scene_name}] ⚠️ Không tìm thấy checkpoint Phase 1 hợp lệ → bỏ qua fine-tune.")
        return 1

    finetune_total_iters = ITERATIONS + FINETUNE_ITERS   # e.g. 35000
    log_file = f"{scene_out}/{scene_name}_finetune.log"
    check_disk_space(scene_name)

    env_prefix = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    cmd = (
        f"{env_prefix}"
        f"CUDA_VISIBLE_DEVICES={gpu_id} python {REPO_DIR}/train.py "
        f"-s {scene_path} "
        f"-m {scene_out} "
        f"-r 1 "                                     # Full resolution
        f"--data_device cpu "
        f"--use_wandb "
        f"--wandb_project bts-digital-twin "
        f"--wandb_entity {WANDB_ENTITY} "
        f"--iterations {finetune_total_iters} "
        f"--lambda_dssim 0.3 "
        # NOTE: --train_test_exp REMOVED — exposure compensation disabled for BTS.
        # NOTE: --exposure_lr_* REMOVED — no exposure optimizer in this pipeline.
        f"--depth_weight_init 0.0 "                  # Tắt depth loss ở Phase 2 fine-tune
        f"--densify_grad_threshold 0.0002 "
        f"--densify_until_iter 0 "                   # QUAN TRỌNG: Tắt hoàn toàn densification
        f"--checkpoint_iterations {finetune_total_iters} "
        f"--save_iterations {finetune_total_iters} "
        f"--disable_viewer "
        f"--start_checkpoint {p1_ckpt}"
    )

    print(f"\n🔬 [Phase 2] Fine-tuning [{scene_name}] tại -r 1 trên GPU {gpu_id} (+{FINETUNE_ITERS} iters)...")
    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT, cwd=REPO_DIR)

    status = "✅ DONE" if result.returncode == 0 else f"❌ FAILED (rc={result.returncode})"
    print(f"  [{scene_name}] Phase 2 {status}")

    if result.returncode != 0:
        print(f"\n{'='*20} FINE-TUNE ERROR LOG ({scene_name}) — Last 30 lines {'='*20}")
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                print("".join(lines[-30:]))
        except Exception as e:
            print(f"Could not read log: {e}")
        print("=" * 60 + "\n")
    else:
        # Cleanup: xoá checkpoint Phase 1 và PLY cũ, chỉ giữ Phase 2
        cleanup_after_train(scene_out, scene_name, final_iter=finetune_total_iters)

    return result.returncode


# ── Xác định WandB entity ─────────────────────────────────────────────────────
import wandb
try:
    viewer = wandb.Api().viewer
    teams  = [t for t in getattr(viewer, 'teams', [])]
    if 'ai_race' in teams or viewer.username == 'ai_race':
        WANDB_ENTITY = 'ai_race'
    else:
        WANDB_ENTITY = viewer.username
    print(f"✅ Đã chốt WandB Entity: {WANDB_ENTITY}")
except Exception:
    WANDB_ENTITY = "ai_race"
    print(f"⚠️ Không thể lấy thông tin, ép dùng mặc định: {WANDB_ENTITY}")


def train_scene(scene_path: str, gpu_id: int) -> tuple:
    """Train một scene. Trả về (scene_name, return_code).

    Chiến lược checkpoint (theo thứ tự ưu tiên kiểm tra):
      1. Tìm checkpoint 30000 hợp lệ → bỏ qua NGAY, không load.
      2. Tìm checkpoint hợp lệ cao nhất < 30000 → resume từ đó.
      3. Không có checkpoint nào → train từ đầu.

    Sau khi train xong 30000:
      - Xác nhận checkpoint 30000 hợp lệ.
      - Xóa checkpoint cũ để giải phóng disk (~500 MB/scene).
    """
    scene_name = os.path.basename(scene_path)
    scene_out  = f"{OUTPUT_DIR}/{scene_name}"
    log_file   = f"{OUTPUT_DIR}/{scene_name}_train.log"
    os.makedirs(scene_out, exist_ok=True)

    def get_iter_from_ckpt(ckpt_path):
        """Trả về số iteration từ tên file checkpoint."""
        try:
            name = os.path.basename(ckpt_path).replace("chkpnt", "").replace(".pth", "")
            return int(name.split("_")[0])
        except:
            return 0

    # ── BƯỚC 1: Kiểm tra checkpoint 30000 trong output dir ────────────────────
    final_ckpt = f"{scene_out}/chkpnt{ITERATIONS}.pth"
    final_ply  = f"{scene_out}/point_cloud/iteration_{ITERATIONS}/point_cloud.ply"

    if is_valid_checkpoint(final_ckpt):
        print(f"  ✅ [{scene_name}] Valid chkpnt{ITERATIONS}.pth found — skipping training")
        if not os.path.exists(final_ply):
            print(f"  ⚠️  [{scene_name}] point_cloud.ply missing — sẽ được tạo khi render")
        return scene_name, 0

    # PLY tồn tại nhưng không có checkpoint (đã cleanup) → cũng skip
    if os.path.exists(final_ply):
        print(f"  ✅ [{scene_name}] point_cloud.ply found — skipping training")
        return scene_name, 0

    # Xóa checkpoint 30000 nếu bị corrupt (tránh resume sai)
    if os.path.exists(final_ckpt):
        print(f"  🗑  [{scene_name}] Corrupt chkpnt{ITERATIONS}.pth detected → removing")
        try:
            os.remove(final_ckpt)
        except Exception as e:
            print(f"    Could not remove: {e}")

    # ── BƯỚC 2: Tìm checkpoint để resume (ưu tiên iter cao nhất < 30000) ──────
    all_local_ckpts = sorted(
        glob.glob(f"{scene_out}/chkpnt*.pth"),
        key=get_iter_from_ckpt
    )

    # Tìm checkpoint từ INPUT_CHECKPOINT_DIR (Kaggle Input dataset)
    if INPUT_CHECKPOINT_DIR:
        input_ckpts = sorted(
            glob.glob(f"{INPUT_CHECKPOINT_DIR}/{scene_name}/chkpnt*.pth") +
            glob.glob(f"{INPUT_CHECKPOINT_DIR}/chkpnt*_{scene_name}.pth") +
            glob.glob(f"{INPUT_CHECKPOINT_DIR}/chkpnt*_{scene_name.lower()}.pth"),
            key=get_iter_from_ckpt
        )
        if input_ckpts:
            best_input = input_ckpts[-1]
            best_iter  = get_iter_from_ckpt(best_input)
            if best_iter >= ITERATIONS and is_valid_checkpoint(best_input):
                print(f"  ✅ [{scene_name}] Valid chkpnt{best_iter} in INPUT_CHECKPOINT_DIR — skipping (no copy)")
                return scene_name, 0
            # Copy các checkpoint hợp lệ < 30000 sang output dir để resume
            for inp_ckpt in input_ckpts:
                if get_iter_from_ckpt(inp_ckpt) < ITERATIONS and is_valid_checkpoint(inp_ckpt):
                    dst = f"{scene_out}/chkpnt{get_iter_from_ckpt(inp_ckpt)}.pth"
                    if not os.path.exists(dst):
                        shutil.copy2(inp_ckpt, dst)
                        print(f"  [{scene_name}] Copied: {os.path.basename(inp_ckpt)} → {os.path.basename(dst)}")
            all_local_ckpts = sorted(glob.glob(f"{scene_out}/chkpnt*.pth"), key=get_iter_from_ckpt)

    # Lọc checkpoint hợp lệ (không bị corrupt, iter < 30000)
    resumable_ckpts = []
    for ckpt in all_local_ckpts:
        it = get_iter_from_ckpt(ckpt)
        if it >= ITERATIONS:
            continue   # checkpoint 30000 đã xử lý ở trên
        if is_valid_checkpoint(ckpt):
            resumable_ckpts.append(ckpt)
        else:
            print(f"  🗑  [{scene_name}] Corrupt checkpoint → removing: {os.path.basename(ckpt)}")
            try:
                os.remove(ckpt)
            except Exception as e:
                print(f"    Could not remove: {e}")

    # Chọn checkpoint cao nhất để resume
    if resumable_ckpts:
        best_ckpt   = resumable_ckpts[-1]
        iter_num    = get_iter_from_ckpt(best_ckpt)
        resume_flag = f"--start_checkpoint {best_ckpt}"
        print(f"  [{scene_name}] Resuming from chkpnt{iter_num} → còn {ITERATIONS - iter_num} iters")
    else:
        resume_flag = ""
        print(f"  [{scene_name}] No valid checkpoint — training from scratch")

    # ── BƯỚC 3: Chạy training ─────────────────────────────────────────────────
    scene_cfg  = get_scene_config(scene_path)
    check_disk_space(scene_name)
    env_prefix = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "

    cmd = (
        f"{env_prefix}"
        f"CUDA_VISIBLE_DEVICES={gpu_id} python {REPO_DIR}/train.py "
        f"-s {scene_path} "
        f"-m {scene_out} "
        f"-r {scene_cfg['resolution']} "
        f"--sh_degree 1 "                    # SH degree 1: giảm 2.5x kích thước checkpoint
        f"--data_device cpu "
        f"--use_wandb "
        f"--wandb_project bts-digital-twin "
        f"--wandb_entity {WANDB_ENTITY} "
        f"--iterations {ITERATIONS} "
        f"--lambda_dssim 0.3 "
        # NOTE: --train_test_exp REMOVED — exposure compensation disabled for BTS.
        # NOTE: --exposure_lr_* REMOVED — exposure optimizer không còn tồn tại.
        f"--depth_weight_init 0.1 "          # Hybrid Depth Scheduler: base weight cho DA-v2
        f"--antialiasing "
        f"--densify_grad_threshold {scene_cfg['densify_grad_threshold']} "
        f"--densify_until_iter {scene_cfg['densify_until_iter']} "
        f"--opacity_reset_interval 2000 "    # Prune Gaussians chết sớn để kiểm soát số lượng điểm
        f"--checkpoint_iterations 15000 {ITERATIONS} "  # Bỏ 7500: checkpoint nhỏ ít cần thiết
        f"--save_iterations {ITERATIONS} "
        f"--disable_viewer "
        f"{resume_flag}"
    )

    print(f"\n🚀 Training [{scene_name}] -r {scene_cfg['resolution']} (≈660×494) on GPU {gpu_id} "
          f"| thresh={scene_cfg['densify_grad_threshold']} densify_until={scene_cfg['densify_until_iter']} ...")
    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT, cwd=REPO_DIR)

    status = "✅ DONE" if result.returncode == 0 else f"❌ FAILED (rc={result.returncode})"
    print(f"  [{scene_name}] {status} — log: {log_file}")

    if result.returncode != 0:
        # In 50 dòng cuối log để debug
        print(f"\n{'='*20} ERROR LOG FOR {scene_name} (Last 50 lines) {'='*20}")
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                print("".join(lines[-50:]))
        except Exception as e:
            print(f"Could not read log: {e}")
        print("=" * 60 + "\n")
    else:
        # ── BƯỚC 4: Sau khi train xong, xác nhận checkpoint 30000 hợp lệ ─────
        # rồi mới xóa checkpoint cũ để giải phóng disk (~500 MB/scene)
        if is_valid_checkpoint(final_ckpt):
            print(f"  ✅ [{scene_name}] chkpnt{ITERATIONS}.pth verified OK — xóa toàn bộ checkpoint cũ")
            total_freed = 0
            for ckpt in glob.glob(f"{scene_out}/chkpnt*.pth"):
                if get_iter_from_ckpt(ckpt) < ITERATIONS:
                    try:
                        sz = os.path.getsize(ckpt)
                        os.remove(ckpt)
                        total_freed += sz
                        print(f"  🗑  [{scene_name}] Deleted: {os.path.basename(ckpt)} ({sz/1024/1024:.0f} MB)")
                    except Exception as e:
                        print(f"  ⚠️  Could not delete {os.path.basename(ckpt)}: {e}")
            if total_freed > 0:
                print(f"  💾 [{scene_name}] Freed {total_freed/1024/1024:.0f} MB from intermediate checkpoints")
        else:
            print(f"  ⚠️  [{scene_name}] chkpnt{ITERATIONS}.pth MISSING or CORRUPT — GIỮ NGUYÊN mọi checkpoint cũ làm backup")

        # Cleanup PLY cũ và Tensorboard logs
        cleanup_after_train(scene_out, scene_name)

    return scene_name, result.returncode


def get_scene_priority(scene_path):
    """Sắp xếp thứ tự train để tối ưu disk và thời gian:
      0 = đã có chkpnt30000 hợp lệ hoặc PLY → skip ngay
      2 = có checkpoint 15000 hợp lệ → resume (ưu tiên cao, gần xong)
      1 = không có gì → train từ đầu
    Ascending → priority 0 xử lý trước (skip nhanh), 2 tiếp theo, 1 cuối.
    """
    scene_name = os.path.basename(scene_path)
    scene_out  = f"{OUTPUT_DIR}/{scene_name}"

    final_ckpt = f"{scene_out}/chkpnt{ITERATIONS}.pth"
    final_ply  = f"{scene_out}/point_cloud/iteration_{ITERATIONS}/point_cloud.ply"
    if os.path.exists(final_ply) or is_valid_checkpoint(final_ckpt):
        return 0   # skip nhanh, trả GPU về queue ngay

    has_15k = bool(glob.glob(f"{scene_out}/chkpnt15000.pth"))
    if INPUT_CHECKPOINT_DIR:
        has_15k = has_15k or bool(
            glob.glob(f"{INPUT_CHECKPOINT_DIR}/{scene_name}/chkpnt15000.pth") +
            glob.glob(f"{INPUT_CHECKPOINT_DIR}/chkpnt15000_{scene_name}.pth") +
            glob.glob(f"{INPUT_CHECKPOINT_DIR}/chkpnt15000_{scene_name.lower()}.pth")
        )
    if has_15k:
        return 2   # resume từ 15000

    return 1   # train từ đầu


print("=" * 60)
print("Sorting scenes by priority...")
all_scenes = sorted(all_scenes, key=get_scene_priority)

print("=" * 60)
print(f"Starting training for {len(all_scenes)} scenes on 2x GPU T4...")
print("=" * 60)

gpu_queue = queue.Queue()
gpu_queue.put(0)
gpu_queue.put(1)

def train_scene_wrapper(scene_path):
    gpu_id = gpu_queue.get()   # block cho đến khi có GPU rảnh
    try:
        return train_scene(scene_path, gpu_id)
    finally:
        gpu_queue.put(gpu_id)   # trả GPU về queue sau khi xong

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        executor.submit(train_scene_wrapper, scene): scene
        for scene in all_scenes
    }
    for future in as_completed(futures):
        scene_name, rc = future.result()

print("\nAll training jobs completed!")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Render test poses (sinh ảnh submission)
# ─────────────────────────────────────────────────────────────────────────────
def render_scene(scene_path: str, gpu_id: int) -> tuple:
    scene_name = os.path.basename(scene_path)
    scene_out  = f"{OUTPUT_DIR}/{scene_name}"
    log_file   = f"{OUTPUT_DIR}/{scene_name}_render.log"

    # Tự động phát hiện iteration cao nhất
    final_iter = get_final_iteration(scene_out)

    # Kiểm tra point_cloud.ply tồn tại trước khi render
    ply_path = f"{scene_out}/point_cloud/iteration_{final_iter}/point_cloud.ply"
    if not os.path.exists(ply_path):
        print(f"  ⚠️  [{scene_name}] Skipping render — point_cloud.ply not found: {ply_path}")
        return scene_name, 1

    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} python {REPO_DIR}/render.py "
        f"-s {scene_path} "
        f"-m {scene_out} "
        f"--skip_train "
        f"--iteration {final_iter}"
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

# Kiểm tra trạng thái training trước khi render
print("\n📊 Training status check:")
trained_scenes = []
failed_scenes  = []
for scene_path in all_scenes:
    scene_name      = os.path.basename(scene_path)
    scene_out_path  = f"{OUTPUT_DIR}/{scene_name}"
    final_iter_check = get_final_iteration(scene_out_path)
    ply_path = f"{scene_out_path}/point_cloud/iteration_{final_iter_check}/point_cloud.ply"
    if os.path.exists(ply_path):
        print(f"  ✅ [{scene_name}] point_cloud.ply found @ iter {final_iter_check}")
        trained_scenes.append(scene_path)
    else:
        print(f"  ❌ [{scene_name}] point_cloud.ply MISSING — will skip render")
        failed_scenes.append(scene_name)

print(f"\n  → {len(trained_scenes)}/{len(all_scenes)} scenes ready to render")
if failed_scenes:
    print(f"  → Skipping: {failed_scenes}")

# Khởi tạo lại GPU queue cho render
render_gpu_queue = queue.Queue()
render_gpu_queue.put(0)
render_gpu_queue.put(1)

def render_scene_wrapper(scene_path):
    gpu_id = render_gpu_queue.get()
    try:
        return render_scene(scene_path, gpu_id)
    finally:
        render_gpu_queue.put(gpu_id)

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        executor.submit(render_scene_wrapper, scene): scene
        # Chỉ render những scene đã train xong (có point_cloud.ply)
        for scene in trained_scenes
    }
    for future in as_completed(futures):
        scene_name, rc = future.result()

print("\nAll renders completed!")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Dong goi submission.zip (smart packer: <= 350 MB)
# ─────────────────────────────────────────────────────────────────────────────
import zipfile, io   # shutil da import o Cell 7
from PIL import Image

SUBMISSION_DIR = "/kaggle/working/submission"
SUBMISSION_ZIP = "/kaggle/working/submission.zip"
ZIP_TARGET_MB  = 350
ZIP_TARGET     = ZIP_TARGET_MB * 1024 * 1024

print("=" * 60)
print("Packaging submission.zip ...")
print("=" * 60)

os.makedirs(SUBMISSION_DIR, exist_ok=True)
missing_scenes = []

for scene_path in all_scenes:
    scene_name  = os.path.basename(scene_path)
    final_iter  = get_final_iteration(f"{OUTPUT_DIR}/{scene_name}")
    render_path = f"{OUTPUT_DIR}/{scene_name}/test/ours_{final_iter}/renders"

    if not os.path.isdir(render_path):
        print(f"  [{scene_name}] Render path not found: {render_path}")
        missing_scenes.append(scene_name)
        continue

    dest = f"{SUBMISSION_DIR}/{scene_name}"
    os.makedirs(dest, exist_ok=True)

    imgs = sorted(glob.glob(f"{render_path}/*.png"))
    for img in imgs:
        shutil.copy(img, dest)

    sample_names = [os.path.basename(p) for p in imgs[:3]]
    print(f"  [{scene_name}] Copied {len(imgs)} images  (samples: {sample_names})")

# -- Ham tien ich ---------------------------------------------------------------

def _collect_images(src_dir):
    pairs = []
    for root, _, files in os.walk(src_dir):
        for f in sorted(files):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                full    = os.path.join(root, f)
                arcname = os.path.relpath(full, src_dir)
                pairs.append((full, arcname))
    return pairs

def _pack_png_lossless(pairs, dst):
    if os.path.exists(dst):
        os.remove(dst)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for full, arcname in pairs:
            zf.write(full, arcname)
    return os.path.getsize(dst)

def _pack_jpeg(pairs, dst, quality):
    """Chuyen PNG -> JPEG in-memory, giu extension .png de khop ten nop bai."""
    n_ok = n_err = 0
    if os.path.exists(dst):
        os.remove(dst)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for full, arcname in pairs:
            try:
                img = Image.open(full).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality, optimize=True, subsampling=0)
                buf.seek(0)
                zf.writestr(arcname, buf.read())
                n_ok += 1
            except Exception as e:
                zf.write(full, arcname)
                n_err += 1
                print(f"    WARNING {arcname}: {e}")
    sz = os.path.getsize(dst)
    print(f"  [JPEG q={quality}] OK={n_ok} ERR={n_err} -> {sz/1024/1024:.1f} MB")
    return sz

# -- Smart packer ---------------------------------------------------------------

pairs   = _collect_images(SUBMISSION_DIR)
raw_mb  = sum(os.path.getsize(p) for p, _ in pairs) / 1024 / 1024
print(f"\nTong {len(pairs)} anh ({raw_mb:.1f} MB raw). Target: <= {ZIP_TARGET_MB} MB\n")

# Buoc 1: Thu PNG lossless
print("Buoc 1: PNG lossless (DEFLATE level=9)...")
zip_size = _pack_png_lossless(pairs, SUBMISSION_ZIP)
zip_mb   = zip_size / 1024 / 1024
print(f"  PNG lossless -> {zip_mb:.1f} MB")

if zip_size <= ZIP_TARGET:
    print(f"OK {zip_mb:.1f} MB <= {ZIP_TARGET_MB} MB -> {SUBMISSION_ZIP}")
else:
    print(f"Qua lon ({zip_mb:.1f} MB). Chuyen sang JPEG...\n")
    for q in [92, 88, 85, 82, 80]:
        print(f"Buoc 2: JPEG quality={q}...")
        zip_size = _pack_jpeg(pairs, SUBMISSION_ZIP, quality=q)
        zip_mb   = zip_size / 1024 / 1024
        if zip_size <= ZIP_TARGET:
            nen_pct = (1 - zip_size / (raw_mb * 1024 * 1024)) * 100
            print(f"OK JPEG q={q}: {zip_mb:.1f} MB <= {ZIP_TARGET_MB} MB (nen {nen_pct:.0f}%)")
            print(f"-> {SUBMISSION_ZIP}")
            break
    else:
        print(f"FAILED: van {zip_mb:.1f} MB > {ZIP_TARGET_MB} MB o JPEG q=80. Giu file nho nhat.")

if missing_scenes:
    print(f"\nMissing renders: {missing_scenes}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 (tuỳ chọn) — Xem preview một số ảnh render
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
from PIL import Image

def preview_scene(scene_name: str, n: int = 3):
    final_iter  = get_final_iteration(f"{OUTPUT_DIR}/{scene_name}")
    render_path = f"{OUTPUT_DIR}/{scene_name}/test/ours_{final_iter}/renders"
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
