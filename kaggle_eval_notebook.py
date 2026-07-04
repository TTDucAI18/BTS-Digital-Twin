# =============================================================================
# BTS Digital Twin — Kaggle Evaluation & Render Notebook
# Chức năng: Đọc checkpoint .pth, sinh ảnh render (test set) và tính điểm (Metrics)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Tải mã nguồn và cài đặt thư viện lõi (Không dùng Wandb)
# ─────────────────────────────────────────────────────────────────────────────
import os, subprocess

def run(cmd, **kwargs):
    """Helper: chạy shell command và in output realtime."""
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True, **kwargs)
    if result.stdout: print(result.stdout)
    if result.stderr: print(result.stderr)
    return result.returncode

REPO_URL  = "https://github.com/TTDucAI18/BTS-Digital-Twin.git"
REPO_DIR  = "/kaggle/working/BTS-Digital-Twin"

print("=" * 60)
print("1. Cloning BTS-Digital-Twin repository...")
print("=" * 60)

if not os.path.exists(REPO_DIR):
    run(f"git clone --recurse-submodules {REPO_URL} {REPO_DIR}")
else:
    run(f"git -C {REPO_DIR} pull origin main")
    run(f"git -C {REPO_DIR} submodule update --init --recursive")

print("=" * 60)
print("2. Installing minimal dependencies & 3DGS submodules...")
print("=" * 60)

# Cài các dependencies cần thiết cho evaluation
run("pip install plyfile tqdm opencv-python lpips \"setuptools<70.0.0\"")

# Compile CUDA submodules
for submod in [
    "submodules/diff-gaussian-rasterization",
    "submodules/simple-knn"
]:
    print(f"\n[Building] {submod} ...")
    rc = run(f"pip install --no-build-isolation -e {submod}", cwd=REPO_DIR)
    status = "✅ OK" if rc == 0 else "❌ FAILED"
    print(f"  → {status}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Cấu hình thư mục và chạy Evaluation + Render
# ─────────────────────────────────────────────────────────────────────────────
import glob

# [1] THƯ MỤC CHỨA ẢNH GỐC & TEST POSES
DATA_DIR = "/kaggle/input/datasets/tdukaggle/ai-race-data/phase1"

# [2] THƯ MỤC CHỨA CÁC CHECKPOINT ĐÃ TRAIN XONG (Thay đổi tên dataset nếu cần)
# Ví dụ: Nếu bạn add output của Kaggle training trước đó làm dataset
CHECKPOINT_DIR = "/kaggle/input/my-models-dataset"

print("=" * 60)
print("Đang tìm kiếm checkpoints...")
print("=" * 60)

# Tìm tất cả file .pth
checkpoints = glob.glob(f"{CHECKPOINT_DIR}/**/*.pth", recursive=True)

# Lọc chỉ lấy checkpoint cuối cùng (VD: chkpnt30000.pth hoặc chkpnt30000_hcm0181.pth)
# Tự động loại bỏ các checkpoint trung gian để đánh giá cho nhanh
final_checkpoints = [ckpt for ckpt in checkpoints if "30000" in os.path.basename(ckpt)]

import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

if not final_checkpoints:
    print(f"⚠️ Không tìm thấy checkpoint nào chứa '30000' trong tên file tại: {CHECKPOINT_DIR}")
    print("Danh sách toàn bộ checkpoint tìm được:", checkpoints)
else:
    print(f"✅ Tìm thấy {len(final_checkpoints)} final checkpoints để đánh giá.")

# Khởi tạo Queue quản lý GPU (T4 Kaggle có 2 GPU)
gpu_queue = queue.Queue()
gpu_queue.put(0)
gpu_queue.put(1)

def eval_worker(ckpt):
    gpu_id = gpu_queue.get()
    ckpt_name = os.path.basename(ckpt)
    try:
        print(f"\n⏳ [GPU {gpu_id}] Đang đánh giá {ckpt_name} ...")
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} python {REPO_DIR}/evaluate_checkpoint.py "
            f"--checkpoint \"{ckpt}\" "
            f"--data_root \"{DATA_DIR}\" "
        )
        
        # Bỏ capture_output để in trực tiếp tiến trình (realtime) ra màn hình
        # Kaggle sẽ tự động gộp log của cả 2 GPU (hơi lộn xộn một chút do tqdm, nhưng bù lại bạn thấy được cả 2 đang chạy)
        subprocess.run(cmd, shell=True)
        
        return f"\n✅ [GPU {gpu_id}] Đã chạy xong: {ckpt_name}"
    finally:
        gpu_queue.put(gpu_id)

if final_checkpoints:
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(eval_worker, ckpt): ckpt for ckpt in final_checkpoints}
        
        for future in as_completed(futures):
            # In ra output nguyên vẹn của từng scene khi nó hoàn thành
            print(future.result())

print("\n" + "=" * 60)
print("✅ HOÀN THÀNH QUÁ TRÌNH ĐÁNH GIÁ TẤT CẢ CHECKPOINTS")
print("=" * 60)
