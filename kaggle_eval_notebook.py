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
import zipfile
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# [1] THƯ MỤC CHỨA ẢNH GỐC & TEST POSES
DATA_DIR = "/kaggle/input/datasets/tdukaggle/ai-race-data/VAI_NVS_DATA/phase1"

# [2] THƯ MỤC CHỨA CÁC CHECKPOINT ĐÃ TRAIN XONG (Thay đổi tên dataset nếu cần)
# Ví dụ: Nếu bạn add output của Kaggle training trước đó làm dataset
CHECKPOINT_DIR = "/kaggle/input/datasets/tdukaggle/ai-race-data"

print("=" * 60)
print("Đang tìm kiếm checkpoints...")
print("=" * 60)

def is_valid_checkpoint(path):
    """Kiểm tra file .pth có phải là ZIP hợp lệ không (tránh lỗi PytorchStreamReader)"""
    try:
        with zipfile.ZipFile(path, 'r') as z:
            bad = z.testzip()  # None nếu tất cả OK
            return bad is None
    except (zipfile.BadZipFile, EOFError, OSError):
        return False

# Tìm tất cả file .pth
checkpoints = glob.glob(f"{CHECKPOINT_DIR}/**/*.pth", recursive=True)

# Lọc checkpoint và kiểm tra tính toàn vẹn (loại bỏ các file bị corrupt)
# Tự động lấy checkpoint có iteration cao nhất hợp lệ cho mỗi scene (phù hợp với cơ chế ngắt sớm)
from collections import defaultdict
scene_ckpts = defaultdict(list)

def get_iter_from_ckpt(ckpt_path):
    try:
        name = os.path.basename(ckpt_path).replace("chkpnt", "").replace(".pth", "")
        return int(name.split("_")[0])
    except:
        return 0

for ckpt in checkpoints:
    parent_dir = os.path.dirname(ckpt)
    scene_ckpts[parent_dir].append(ckpt)

final_checkpoints = []
for parent_dir, ckpts in scene_ckpts.items():
    # Sắp xếp checkpoint theo iteration giảm dần
    ckpts_sorted = sorted(ckpts, key=get_iter_from_ckpt, reverse=True)
    for ckpt in ckpts_sorted:
        if is_valid_checkpoint(ckpt):
            final_checkpoints.append(ckpt)
            break # Lấy checkpoint có iteration cao nhất hợp lệ cho scene này
        else:
            print(f"❌ CẢNH BÁO: Checkpoint bị lỗi (corrupted) - bỏ qua: {ckpt}")

if not final_checkpoints:
    print(f"⚠️ Không tìm thấy checkpoint hợp lệ nào tại: {CHECKPOINT_DIR}")
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
            f"--save_renders_dir \"/kaggle/working/eval_renders\" "
        )

        # Bỏ capture_output để in trực tiếp tiến trình (realtime) ra màn hình
        result = subprocess.run(cmd, shell=True)
        status = "✅ OK" if result.returncode == 0 else f"❌ FAILED (rc={result.returncode})"
        return f"\n{status} [GPU {gpu_id}] Hoàn thành: {ckpt_name}"
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


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Trực quan hoá kết quả (Visualization)
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

RENDERS_DIR = "/kaggle/working/eval_renders"

if not os.path.exists(RENDERS_DIR):
    print(f"⚠️ Chưa có ảnh render nào trong {RENDERS_DIR}. Hãy chắc chắn rằng bạn đã chạy Cell 2 thành công.")
else:
    print("=" * 60)
    print("Trực quan hoá kết quả (Ground Truth vs Render)...")
    print("=" * 60)
    
    scenes = [d for d in os.listdir(RENDERS_DIR) if os.path.isdir(os.path.join(RENDERS_DIR, d))]
    if not scenes:
        print(f"⚠️ Không tìm thấy scene nào trong {RENDERS_DIR}.")
    else:
        for scene in scenes:
            scene_render_dir = os.path.join(RENDERS_DIR, scene)
            
            # Lấy danh sách ảnh render
            images = [img for img in os.listdir(scene_render_dir) if img.lower().endswith((".png", ".jpg", ".jpeg"))]
            if not images: continue
            
            # Chọn ngẫu nhiên tối đa 2 ảnh để hiển thị cho mỗi scene
            num_samples = min(2, len(images))
            sample_images = random.sample(images, num_samples)
            
            # Xử lý subplot dimensions
            fig, axes = plt.subplots(num_samples, 2, figsize=(16, 6 * num_samples))
            fig.suptitle(f"SCENE: {scene.upper()}", fontsize=18, fontweight="bold", y=1.02)
            
            for i, img_name in enumerate(sample_images):
                render_path = os.path.join(scene_render_dir, img_name)
                
                # Tìm đường dẫn ảnh Ground Truth tương ứng
                gt_path = None
                for d in os.listdir(DATA_DIR):
                    if d.lower() == scene.lower():
                        gt_path = os.path.join(DATA_DIR, d, "test", "images", img_name)
                        break
                
                # Xử lý indexing cho array axes (1D vs 2D)
                if num_samples == 1:
                    ax_gt = axes[0]
                    ax_render = axes[1]
                else:
                    ax_gt = axes[i, 0]
                    ax_render = axes[i, 1]
                
                # Plot Ground Truth
                if gt_path and os.path.exists(gt_path):
                    ax_gt.imshow(mpimg.imread(gt_path))
                else:
                    ax_gt.text(0.5, 0.5, 'GT Missing', horizontalalignment='center', verticalalignment='center', fontsize=20)
                ax_gt.set_title(f"Ground Truth ({img_name})", fontsize=14)
                ax_gt.axis("off")
                
                # Plot Render Output
                ax_render.imshow(mpimg.imread(render_path))
                ax_render.set_title(f"3DGS Render ({img_name})", fontsize=14)
                ax_render.axis("off")
                
            plt.tight_layout()
            plt.show()
