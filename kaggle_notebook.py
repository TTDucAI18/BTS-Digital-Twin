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
# CELL 7 — Chạy Training Multi-GPU (queue-based: scene nào xong sớm thì GPU đó nhận scene tiếp)
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, queue, shutil, time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_DIR   = "/kaggle/working/BTS-Digital-Twin"
OUTPUT_DIR = "/kaggle/working/output"
ITERATIONS      = 30_000  # Train đầy đủ 30k iters tại -r 1
FINETUNE_ITERS  = 0       # Fine-tune TẮT: thực nghiệm cho thấy Phase 2 gây regression
KAGGLE_TIME_LIMIT_H  = 10.0         # Ngưỡng an toàn (12h limit - 2h buffer)
KAGGLE_SESSION_START = time.time()  # Bắt đầu tính giờ ngay khi chạy Cell 7

# ĐƯỜNG DẪN TỚI CHECKPOINT TỪ KAGGLE INPUT (nếu có). 
# Ví dụ: "/kaggle/input/my-models-dataset" (bên trong phải chứa các thư mục con mang tên scene như HCM0193, hcm0031...)
# Bỏ trống "" nếu chỉ muốn tự động resume từ output hiện tại.
INPUT_CHECKPOINT_DIR = ""

def check_disk_space(label: str = ""):
    """In dung lượng còn trống trên /kaggle/working."""
    total, used, free = shutil.disk_usage("/kaggle/working")
    free_gb  = free  / (1024 ** 3)
    total_gb = total / (1024 ** 3)
    tag = f"[{label}] " if label else ""
    flag = "⚠️ LOW DISK" if free_gb < 3 else "💾"
    print(f"  {flag} {tag}Disk: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
    return free_gb

def cleanup_after_train(scene_out: str, scene_name: str, final_iter: int = None):
    """Giải phóng disk sau khi train xong một scene.
    - Xoá checkpoint trung gian (giữ lại checkpoint mới nhất)
    - Xoá thư mục point_cloud cũ (giữ lại iteration mới nhất)
    final_iter: iteration cần giữ lại. None = tự động detect từ thư mục.
    """
    if final_iter is None:
        final_iter = get_final_iteration(scene_out)
    freed = 0

    # 1. Xoá checkpoint trung gian, chỉ giữ checkpoint mới nhất
    keep_ckpt = f"{scene_out}/chkpnt{final_iter}.pth"
    for ckpt in glob.glob(f"{scene_out}/chkpnt*.pth"):
        if ckpt != keep_ckpt:
            try:
                size = os.path.getsize(ckpt)
                os.remove(ckpt)
                freed += size
                print(f"    🗑  Removed checkpoint: {os.path.basename(ckpt)}")
            except Exception as e:
                print(f"    ⚠️  Could not remove {ckpt}: {e}")

    # Xóa file Tensorboard logs (nếu vô tình sinh ra)
    for tfevent in glob.glob(f"{scene_out}/events.out.tfevents*"):
        try:
            size = os.path.getsize(tfevent)
            os.remove(tfevent)
            freed += size
        except: pass

    # Xoá bớt thư mục offline run của wandb để nhẹ máy
    wandb_dir = "/kaggle/working/wandb"
    if os.path.exists(wandb_dir):
        for run_dir in os.listdir(wandb_dir):
            if run_dir.startswith("run-") or run_dir.startswith("offline-"):
                try:
                    shutil.rmtree(os.path.join(wandb_dir, run_dir))
                except: pass

    # 2. Xoá point_cloud của iteration cũ, chỉ giữ iteration mới nhất
    pc_base = f"{scene_out}/point_cloud"
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

    print(f"  [{scene_name}] 🧹 Cleanup freed: {freed / (1024**3):.2f} GB")
    check_disk_space(scene_name)

def is_valid_checkpoint(path: str) -> bool:
    """Kiểm tra checkpoint có phải ZIP hợp lệ không (PyTorch .pth là zip archive).
    File bị truncate do disk-full sẽ thiếu magic bytes và central directory.
    """
    import zipfile
    try:
        with zipfile.ZipFile(path, 'r') as zf:
            bad = zf.testzip()  # None nếu tất cả OK
            return bad is None
    except (zipfile.BadZipFile, EOFError, OSError):
        return False

def has_time_remaining(hours_needed: float) -> bool:
    """Kiểm tra session còn đủ thời gian để chạy thêm bước không.
    Tránh bị Kaggle cắt giữa chừng (timeout 12h).
    """
    elapsed_h = (time.time() - KAGGLE_SESSION_START) / 3600
    remaining_h = KAGGLE_TIME_LIMIT_H - elapsed_h
    if remaining_h < hours_needed:
        print(f"  ⏱ Thời gian còn lại: {remaining_h:.1f}h < cần {hours_needed:.1f}h → bỏ qua bước này.")
        return False
    print(f"  ⏱ Thời gian còn lại: {remaining_h:.1f}h → đủ để chạy thêm {hours_needed:.1f}h")
    return True

def get_final_iteration(scene_out: str) -> int:
    """Tìm iteration cao nhất có sẵn trong thư mục point_cloud.
    Tự động nhận ra Phase 2 (35000) nếu đã chạy, fallback về Phase 1 (30000).
    """
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
    Ảnh dataset AI Race đã được scale 1/4 xuống 1320×989 → dùng -r 1 an toàn với T4 16GB.
    Heuristic dựa trên data drone UAV HCM/HNI dataset.
    """
    # BTS structure: scene_path/train/images (không phải scene_path/images)
    img_dir = os.path.join(scene_path, "train", "images")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(scene_path, "images")  # fallback cho cấu trúc khác
    if os.path.isdir(img_dir):
        n_imgs = len([f for f in os.listdir(img_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    else:
        n_imgs = 240  # Không detect được → assume trung bình
    print(f"  [Config] Số ảnh train: {n_imgs}")

    # Tất cả scene đều dùng -r 1 vì ảnh đã scale 1/4 → 1320×989 fit T4 16GB
    # densify_until_iter 20000 (từ 15000) → densify lâu hơn, scene phức tạp được bao phủ đủ
    if n_imgs <= 120:
        # Scene nhỏ (HCM1439: 103 ảnh): densify đầy đủ, threshold thấp hơn để bắt chi tiết
        return {"resolution": 1, "densify_until_iter": 20000, "densify_grad_threshold": 0.0001}
    elif n_imgs <= 220:
        # Scene vừa (HNI0265: 205, HNI0437: 224 ảnh)
        return {"resolution": 1, "densify_until_iter": 20000, "densify_grad_threshold": 0.0002}
    else:
        # Scene đầy đủ (240 ảnh): threshold chuẩn 3DGS
        return {"resolution": 1, "densify_until_iter": 20000, "densify_grad_threshold": 0.0002}

def finetune_scene(scene_path: str, scene_out: str, gpu_id: int) -> int:
    """Phase 2: Fine-tune tại -r 1 (full resolution) sau khi Phase 1 xong.

    Tại sao an toàn về VRAM:
    - Không chạy densification (densify_until_iter=0) → không có densification
      stats tensor → VRAM tiết kiệm đáng kể so với Phase 1.
    - Gaussians đã ở đúng vị trí từ Phase 1, Phase 2 chỉ tinh chỉnh
      màu sắc/opacity tại full-res → gains lớn về PSNR/SSIM.
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

    finetune_total_iters = ITERATIONS + FINETUNE_ITERS  # e.g. 35000
    log_file = f"{scene_out}/{scene_name}_finetune.log"
    check_disk_space(scene_name)

    env_prefix = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    cmd = (
        f"{env_prefix}"
        f"CUDA_VISIBLE_DEVICES={gpu_id} python {REPO_DIR}/train.py "
        f"-s {scene_path} "
        f"-m {scene_out} "
        f"-r 1 "                                    # Full resolution!
        f"--use_wandb "
        f"--wandb_project bts-digital-twin "
        f"--wandb_entity {WANDB_ENTITY} "
        f"--iterations {finetune_total_iters} "
        f"--lambda_dssim 0.5 "
        f"--train_test_exp "
        f"--densify_grad_threshold 0.0002 "
        f"--densify_until_iter 0 "                  # QUAN TRỌNG: Tắt hoàn toàn densification
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

# Ưu tiên sử dụng entity (team) "ai_race" nếu tài khoản có quyền truy cập, nếu không dùng username
import wandb
try:
    viewer = wandb.Api().viewer
    teams = [t for t in getattr(viewer, 'teams', [])]
    if 'ai_race' in teams or viewer.username == 'ai_race':
        WANDB_ENTITY = 'ai_race'
    else:
        WANDB_ENTITY = viewer.username
    print(f"✅ Đã chốt WandB Entity: {WANDB_ENTITY}")
except:
    WANDB_ENTITY = "ai_race"
    print(f"⚠️ Không thể lấy thông tin, ép dùng mặc định: {WANDB_ENTITY}")

def train_scene(scene_path: str, gpu_id: int) -> str:
    scene_name = os.path.basename(scene_path)
    scene_out  = f"{OUTPUT_DIR}/{scene_name}"
    log_file   = f"{OUTPUT_DIR}/{scene_name}_train.log"
    os.makedirs(scene_out, exist_ok=True)

    # Skip nếu đã train xong (point_cloud.ply tồn tại)
    final_ply = f"{scene_out}/point_cloud/iteration_{ITERATIONS}/point_cloud.ply"
    if os.path.exists(final_ply):
        print(f"  ✅ [{scene_name}] Already trained (point_cloud.ply found) — skipping")
        return scene_name, 0

    # Hàm phụ để lấy số iteration từ tên file
    # Hỗ trợ 2 format:
    #   - Chuẩn:  chkpnt7000.pth
    #   - Flat:   chkpnt7000_hcm0181.pth  (tên file có scene name)
    def get_iter_from_ckpt(ckpt_path):
        try:
            name = os.path.basename(ckpt_path).replace("chkpnt", "").replace(".pth", "")
            # Nếu có hậu tố _scenename thì bỏ phần đó đi: "7000_hcm0181" → "7000"
            name = name.split("_")[0]
            return int(name)
        except:
            return 0

    # Auto-resume: tìm checkpoint theo thứ tự ưu tiên
    ckpts = []

    if INPUT_CHECKPOINT_DIR:
        # Ưu tiên 1: subfolder chuẩn → INPUT_CHECKPOINT_DIR/HCM0181/chkpnt*.pth
        subdir_ckpts = sorted(
            glob.glob(f"{INPUT_CHECKPOINT_DIR}/{scene_name}/chkpnt*.pth"),
            key=get_iter_from_ckpt
        )
        flat_pattern_exact = f"{INPUT_CHECKPOINT_DIR}/chkpnt*_{scene_name}.pth"
        flat_pattern_lower = f"{INPUT_CHECKPOINT_DIR}/chkpnt*_{scene_name.lower()}.pth"
        
        flat_ckpts = []
        for p in set(glob.glob(flat_pattern_exact) + glob.glob(flat_pattern_lower)):
            flat_ckpts.append(p)
        flat_ckpts = sorted(flat_ckpts, key=get_iter_from_ckpt)

        if subdir_ckpts:
            max_iter = get_iter_from_ckpt(subdir_ckpts[-1])
            if max_iter >= ITERATIONS:
                print(f"  ✅ [{scene_name}] Checkpoint @ iter {max_iter} found in input — skipping completely, NO disk copy")
                return scene_name, 0
            ckpts = subdir_ckpts
            print(f"  [{scene_name}] Found {len(ckpts)} checkpoint(s) in subfolder")
        elif flat_ckpts:
            max_iter = get_iter_from_ckpt(flat_ckpts[-1])
            if max_iter >= ITERATIONS:
                print(f"  ✅ [{scene_name}] Checkpoint @ iter {max_iter} found in input — skipping completely, NO disk copy")
                return scene_name, 0
            for flat_ckpt in flat_ckpts:
                iter_num = get_iter_from_ckpt(flat_ckpt)
                dst = f"{scene_out}/chkpnt{iter_num}.pth"
                if not os.path.exists(dst):
                    shutil.copy2(flat_ckpt, dst)
                    print(f"  [{scene_name}] Copied flat checkpoint: {os.path.basename(flat_ckpt)} → {os.path.basename(dst)}")
            ckpts = sorted(glob.glob(f"{scene_out}/chkpnt*.pth"), key=get_iter_from_ckpt)

    if not ckpts:
        ckpts = sorted(glob.glob(f"{scene_out}/chkpnt*.pth"), key=get_iter_from_ckpt)

    valid_ckpts = []
    for ckpt in ckpts:
        if is_valid_checkpoint(ckpt):
            valid_ckpts.append(ckpt)
        else:
            print(f"  ⚠️  [{scene_name}] Corrupt checkpoint detected, removing: {os.path.basename(ckpt)}")
            try:
                os.remove(ckpt)
            except Exception as e:
                print(f"    Could not remove: {e}")
    ckpts = valid_ckpts

    resume_flag = f"--start_checkpoint {ckpts[-1]}" if ckpts else ""
    if resume_flag:
        iter_num = get_iter_from_ckpt(ckpts[-1])
        remaining = ITERATIONS - iter_num

        if iter_num >= ITERATIONS:
            if os.path.exists(final_ply):
                print(f"  ✅ [{scene_name}] Checkpoint @ iter {iter_num} + point_cloud.ply found — skipping")
            else:
                print(f"  ⚠️  [{scene_name}] Checkpoint @ iter {iter_num} found but point_cloud.ply MISSING")
                print(f"       → Training sẽ chạy 0 iters, nhưng point_cloud.ply sẽ không được tạo tự động")
                print(f"       → Cần chạy render để dùng checkpoint này trực tiếp, hoặc xoá checkpoint để train lại")
            return scene_name, 0

        print(f"  [{scene_name}] Resuming from iter {iter_num} → còn {remaining} iters nữa")
    else:
        print(f"  [{scene_name}] No checkpoint found — training from scratch")

    # Phase 1: Train nền với config adaptive theo số ảnh của scene
    # get_scene_config(): tự động chọn resolution và densify config phù hợp
    # --train_test_exp: per-image appearance embedding cho drone auto-exposure
    scene_cfg = get_scene_config(scene_path)
    check_disk_space(scene_name)
    env_prefix = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    cmd = (
        f"{env_prefix}"
        f"CUDA_VISIBLE_DEVICES={gpu_id} python {REPO_DIR}/train.py "
        f"-s {scene_path} "
        f"-m {scene_out} "
        f"-r {scene_cfg['resolution']} "
        f"--use_wandb "
        f"--wandb_project bts-digital-twin "
        f"--wandb_entity {WANDB_ENTITY} "
        f"--iterations {ITERATIONS} "
        f"--lambda_dssim 0.2 "
        f"--train_test_exp "
        f"--antialiasing "
        f"--densify_grad_threshold {scene_cfg['densify_grad_threshold']} "
        f"--densify_until_iter {scene_cfg['densify_until_iter']} "
        f"--checkpoint_iterations 15000 {ITERATIONS} "
        f"--save_iterations {ITERATIONS} "
        f"--disable_viewer "
        f"{resume_flag}"
    )

    print(f"\n🚀 [Phase 1] Training [{scene_name}] -r {scene_cfg['resolution']} on GPU {gpu_id} ...")
    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT, cwd=REPO_DIR)

    status = "✅ DONE" if result.returncode == 0 else f"❌ FAILED (rc={result.returncode})"
    print(f"  [{scene_name}] Phase 1 {status} — log: {log_file}")

    if result.returncode != 0:
        print(f"\n{'='*20} ERROR LOG FOR {scene_name} (Last 50 lines) {'='*20}")
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                print("".join(lines[-50:]))
        except Exception as e:
            print(f"Could not read log: {e}")
        print("=" * 60 + "\n")
    else:
        # Cleanup Phase 1 intermediate files để giải phóng disk
        cleanup_after_train(scene_out, scene_name, final_iter=ITERATIONS)
        # Phase 2: Fine-tune tại full resolution nếu còn đủ thời gian
        finetune_scene(scene_path, scene_out, gpu_id)

    return scene_name, result.returncode


def get_scene_priority(scene_path):
    scene_name = os.path.basename(scene_path)
    final_ply = f"{OUTPUT_DIR}/{scene_name}/point_cloud/iteration_{ITERATIONS}/point_cloud.ply"
    if os.path.exists(final_ply):
        return 3 # Already fully trained
    
    has_ckpt = False
    if INPUT_CHECKPOINT_DIR:
        if glob.glob(f"{INPUT_CHECKPOINT_DIR}/{scene_name}/chkpnt*.pth") or \
           glob.glob(f"{INPUT_CHECKPOINT_DIR}/chkpnt*_{scene_name}.pth") or \
           glob.glob(f"{INPUT_CHECKPOINT_DIR}/chkpnt*_{scene_name.lower()}.pth"):
            has_ckpt = True
            
    if glob.glob(f"{OUTPUT_DIR}/{scene_name}/chkpnt*.pth"):
        has_ckpt = True
        
    return 2 if has_ckpt else 1

print("=" * 60)
print("Sorting scenes by priority (scenes without checkpoints first)...")
all_scenes = sorted(all_scenes, key=get_scene_priority)

# Chạy song song: queue-based GPU scheduler (GPU nào rảnh sẽ nhận scene tiếp theo)
print("=" * 60)
print(f"Starting training for {len(all_scenes)} scenes on 2x GPU T4...")
print("=" * 60)

gpu_queue = queue.Queue()
gpu_queue.put(0)
gpu_queue.put(1)

def train_scene_wrapper(scene_path):
    gpu_id = gpu_queue.get()  # block cho đến khi có GPU rảnh
    try:
        return train_scene(scene_path, gpu_id)
    finally:
        gpu_queue.put(gpu_id)  # trả GPU về queue sau khi xong

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
def render_scene(scene_path: str, gpu_id: int) -> str:
    scene_name = os.path.basename(scene_path)
    scene_out  = f"{OUTPUT_DIR}/{scene_name}"
    log_file   = f"{OUTPUT_DIR}/{scene_name}_render.log"

    # Tự động phát hiện iteration cao nhất (Phase 2 nếu đã chạy, Phase 1 nếu không)
    final_iter = get_final_iteration(scene_out)

    # Kiểm tra point_cloud.ply tồn tại trước khi render để tránh crash
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

# Kiểm tra trạng thái training trước khi render (hỗ trợ cả Phase 1 và Phase 2)
print("\n📊 Training status check:")
trained_scenes = []
failed_scenes = []
for scene_path in all_scenes:
    scene_name = os.path.basename(scene_path)
    scene_out_path = f"{OUTPUT_DIR}/{scene_name}"
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

# Khởi tạo lại queue cho quá trình render
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
        # BUG FIX: Chỉ render những scene đã train xong (có point_cloud.ply)
        for scene in trained_scenes
    }
    for future in as_completed(futures):
        scene_name, rc = future.result()

print("\nAll renders completed!")



# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Đóng gói submission.zip
# ─────────────────────────────────────────────────────────────────────────────
import zipfile  # shutil đã import ở Cell 7

SUBMISSION_DIR = "/kaggle/working/submission"
SUBMISSION_ZIP = "/kaggle/working/submission.zip"

print("=" * 60)
print("Packaging submission.zip ...")
print("=" * 60)

os.makedirs(SUBMISSION_DIR, exist_ok=True)
missing_scenes = []

for scene_path in all_scenes:
    scene_name  = os.path.basename(scene_path)
    # Tự động dùng iteration cao nhất (Phase 2 nếu có, Phase 1 nếu không)
    final_iter  = get_final_iteration(f"{OUTPUT_DIR}/{scene_name}")
    render_path = f"{OUTPUT_DIR}/{scene_name}/test/ours_{final_iter}/renders"

    if not os.path.isdir(render_path):
        print(f"  ⚠️  [{scene_name}] Render path not found: {render_path}")
        missing_scenes.append(scene_name)
        continue

    dest = f"{SUBMISSION_DIR}/{scene_name}"
    os.makedirs(dest, exist_ok=True)

    imgs = sorted(glob.glob(f"{render_path}/*.png"))
    for img in imgs:
        shutil.copy(img, dest)

    # Log mẫu tên ảnh để xác nhận khớp với test_poses.csv
    sample_names = [os.path.basename(p) for p in imgs[:3]]
    print(f"  ✅ [{scene_name}] Copied {len(imgs)} images  (samples: {sample_names})")

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
