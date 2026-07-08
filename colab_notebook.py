# =============================================================================
# BTS Digital Twin — Colab Notebook (A100 40GB VRAM Optimized)
# Khắc phục lỗi hình học, tối đa hóa chất lượng và sinh submission.
# =============================================================================

import os
import subprocess
import glob
import shutil
import time
import threading
from tqdm import tqdm
from PIL import Image

# 1. SETUP MÔI TRƯỜNG & ĐƯỜNG DẪN
print("=" * 60)
print("1. SETUP MÔI TRƯỜNG & GOOGLE DRIVE")
print("=" * 60)

# Cố gắng mount Google Drive nếu đang ở trên Colab
try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    print("✅ Đã kết nối Google Drive.")
    IN_COLAB = True
except ImportError:
    print("⚠️ Không chạy trên Colab. Sử dụng đường dẫn Local.")
    IN_COLAB = False

# Đường dẫn Data (Hỗ trợ cả Local Windows và Colab)
if IN_COLAB:
    DATA_DIR = "/content/drive/MyDrive/data/phase1/private_set1"
    OUTPUT_DIR = "/content/output"
    SUBMISSION_DIR = "/content/submission"
    DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/3DGS_Checkpoints_A100"
    REPO_DIR = "/content/BTS-Digital-Twin"
else:
    DATA_DIR = r"D:\ai_race_2026\data\phase1\private_set1"
    OUTPUT_DIR = r"D:\ai_race_2026\output"
    SUBMISSION_DIR = r"D:\ai_race_2026\submission"
    DRIVE_OUTPUT_DIR = r"D:\ai_race_2026\3DGS_Checkpoints_A100"
    REPO_DIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)

# Tìm tất cả 8 scenes trong private_set1
all_scenes = sorted([p for p in glob.glob(os.path.join(DATA_DIR, "*")) if os.path.isdir(p)])
print(f"📁 Tìm thấy {len(all_scenes)} scenes trong {DATA_DIR}")

def run(cmd, **kwargs):
    print(f"➜ Running: {cmd}")
    result = subprocess.run(cmd, shell=True, **kwargs)
    if result.returncode != 0:
        print(f"❌ Lệnh thất bại với mã lỗi {result.returncode}")
    return result.returncode

# Tự động clone repo và cài đặt trên Colab
if IN_COLAB:
    WHEEL_DIR = "/content/drive/MyDrive/wheels"
    os.makedirs(WHEEL_DIR, exist_ok=True)
    
    def install_or_build_module(module_name, module_path, wheel_prefix):
        wheels = glob.glob(os.path.join(WHEEL_DIR, f"{wheel_prefix}*.whl"))
        if wheels:
            print(f"📦 Đã tìm thấy wheel cho {module_name} trên Drive. Tiến hành cài đặt...")
            run(f"pip install -q {wheels[0]}")
        else:
            print(f"⚙️ Đang biên dịch và tạo wheel cho {module_name} (lưu vào Drive)...")
            if module_name == "fused-ssim":
                print(f"⚙️ Cài đặt trực tiếp {module_name} bằng setup.py (bỏ qua pip)...")
                ret2 = run(f"cd {module_path} && python setup.py install > install_{module_name}.log 2>&1")
                if ret2 != 0:
                    print(f"⚠️ Install thất bại cho {module_name}. Log lỗi:")
                    try:
                        with open(f"{module_path}/install_{module_name}.log", "r") as f:
                            print(f.read())
                    except:
                        pass
                return
                
            ret = run(f"pip wheel -v {module_path} --no-build-isolation -w {WHEEL_DIR} > build_{module_name}.log 2>&1")
            if ret != 0:
                print(f"⚠️ Build wheel thất bại cho {module_name}. Log lỗi:")
                try:
                    with open(f"build_{module_name}.log", "r") as f:
                        print(f.read())
                except:
                    pass
                
            new_wheels = glob.glob(os.path.join(WHEEL_DIR, f"{wheel_prefix}*.whl"))
            if new_wheels:
                run(f"pip install --no-build-isolation -q {new_wheels[0]}")
            else:
                print(f"⚠️ Không thể tạo wheel cho {module_name}, cài đặt thông thường.")
                ret2 = run(f"pip install -v -e {module_path} --no-build-isolation > install_{module_name}.log 2>&1")
                if ret2 != 0:
                    print(f"⚠️ Install thất bại cho {module_name}. Log lỗi:")
                    try:
                        with open(f"install_{module_name}.log", "r") as f:
                            print(f.read())
                    except:
                        pass

    if not os.path.exists(REPO_DIR):
        print("\n" + "=" * 60)
        print("📥 ĐANG TẢI MÃ NGUỒN VÀ CÀI ĐẶT CUDA KERNELS...")
        print("=" * 60)
        run(f"git clone --recursive https://github.com/TTDucAI18/BTS-Digital-Twin.git {REPO_DIR}")
        run(f"pip install -q plyfile tqdm wandb ninja")
        
        # Xóa pyproject.toml của fused-ssim để ép pip dùng legacy setup.py (tránh lỗi PEP 517 metadata trên Colab)
        run(f"rm -f {REPO_DIR}/submodules/fused-ssim/pyproject.toml")
        
        install_or_build_module("diff-gaussian-rasterization", f"{REPO_DIR}/submodules/diff-gaussian-rasterization", "diff_gaussian_rasterization")
        install_or_build_module("simple-knn", f"{REPO_DIR}/submodules/simple-knn", "simple_knn")
        install_or_build_module("fused-ssim", f"{REPO_DIR}/submodules/fused-ssim", "fused_ssim")
        
        print("✅ Cài đặt môi trường hoàn tất!")
    else:
        print("🔄 Cập nhật mã nguồn mới nhất...")
        run(f"cd {REPO_DIR} && git pull")
        run(f"cd {REPO_DIR} && git submodule update --init --recursive")
        
        # Xóa pyproject.toml trong trường hợp cập nhật lại source
        run(f"rm -f {REPO_DIR}/submodules/fused-ssim/pyproject.toml")
        
        install_or_build_module("diff-gaussian-rasterization", f"{REPO_DIR}/submodules/diff-gaussian-rasterization", "diff_gaussian_rasterization")
        install_or_build_module("simple-knn", f"{REPO_DIR}/submodules/simple-knn", "simple_knn")
        install_or_build_module("fused-ssim", f"{REPO_DIR}/submodules/fused-ssim", "fused_ssim")

print("\n" + "=" * 60)
print("2. BẮT ĐẦU HUẤN LUYỆN VÀ RENDER (TỐI ƯU CHO A100 40GB)")
print("=" * 60)

class GPUMonitor(threading.Thread):
    def __init__(self, pbar):
        super().__init__()
        self.pbar = pbar
        self.daemon = True
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                res = subprocess.check_output(
                    "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits", 
                    shell=True).decode('utf-8').strip()
                # Lấy thông số GPU đầu tiên
                util, mem_used, mem_total = res.split('\n')[0].split(', ')
                self.pbar.set_postfix_str(f"VRAM: {mem_used}/{mem_total} MB | Util: {util}%")
            except:
                pass
            time.sleep(2)

pbar = tqdm(all_scenes, desc="Tổng tiến trình các Scenes", position=0, leave=True)
monitor = GPUMonitor(pbar)
monitor.start()

for scene in pbar:
    scene_name = os.path.basename(scene)
    scene_out = os.path.join(OUTPUT_DIR, scene_name)
    drive_scene_out = os.path.join(DRIVE_OUTPUT_DIR, scene_name)
    print(f"\n🚀 XỬ LÝ SCENE: {scene_name}")

    # Phục hồi dữ liệu từ Drive nếu có (Resume training)
    if not os.path.exists(scene_out) and os.path.exists(drive_scene_out):
        print(f"  🔄 Đang phục hồi checkpoint từ Google Drive...")
        shutil.copytree(drive_scene_out, scene_out)

    is_trained = os.path.exists(os.path.join(scene_out, "point_cloud", "iteration_30000", "point_cloud.ply"))
    
    if not is_trained:
        print(f"  [1/2] Đang huấn luyện 3D Gaussian Splatting (A100 Optimized)...")
        # --- CÁC THAM SỐ TỐI ƯU CHO A100 (Chất lượng cực cao) ---
        # 1. Không downscale ảnh (--resolution -1)
        # 2. Sinh nhiều hạt Gaussians hơn: densify_grad_threshold giảm từ 0.0002 -> 0.0001
        # 3. Kéo dài thời gian sinh hạt: densify_until_iter lên 20000
        # 4. Giảm chu kỳ sinh hạt: densification_interval xuống 50
        cmd_train = (
            f"bash -c \"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; set -o pipefail; python {os.path.join(REPO_DIR, 'train.py')} "
            f"-s {scene} "
            f"-m {scene_out} "
            f"--iterations 30000 "
            f"--save_iterations 30000 "
            f"--densify_until_iter 20000 "
            f"--densification_interval 50 "
            f"--densify_grad_threshold 0.0001 "
            f"--opacity_reset_interval 5000 "
            f"--lambda_dssim 0.2 "
            f"--sh_degree 3 "
            f"--disable_viewer "
            f"2>&1 | tee train_scene.log\""
        )
        ret_train = run(cmd_train)
        if ret_train != 0:
            print(f"  ❌ Huấn luyện 3DGS thất bại. Bỏ qua scene này. Chi tiết lỗi:")
            try:
                with open("train_scene.log", "r") as f:
                    lines = f.readlines()
                    print("".join(lines[-50:]))
            except:
                pass
            continue
        print(f"  ✅ Huấn luyện 3DGS hoàn tất ở iter 30000.")
        
        # Lưu Checkpoint vĩnh viễn
        if not os.path.exists(drive_scene_out):
            shutil.copytree(scene_out, drive_scene_out)
        else:
            # dirs_exist_ok=True giúp ghi đè nội dung mới lên thư mục đã tồn tại
            shutil.copytree(scene_out, drive_scene_out, dirs_exist_ok=True)
    else:
        print(f"  ✅ Scene {scene_name} đã được huấn luyện đủ 30000 iterations từ trước.")

    # Render Test Views
    render_path = os.path.join(scene_out, "test", "ours_30000", "renders")
    if not os.path.exists(render_path) or len(glob.glob(f"{render_path}/*.*")) == 0:
        print(f"  [2/2] Đang render các góc nhìn Test (Novel Views)...")
        cmd_render = (
            f"bash -c \"set -o pipefail; python {os.path.join(REPO_DIR, 'render.py')} "
            f"-s {scene} "
            f"-m {scene_out} "
            f"--skip_train "
            f"--iteration 30000 "
            f"--sh_degree 3 "
            f"2>&1 | tee render_scene.log\""
        )
        ret_render = run(cmd_render)
        if ret_render != 0:
            print(f"  ❌ Render thất bại. Bỏ qua copy submission. Chi tiết lỗi:")
            try:
                with open("render_scene.log", "r") as f:
                    lines = f.readlines()
                    print("".join(lines[-50:]))
            except:
                pass
            continue
        print(f"  ✅ Render Novel Views thành công.")
    else:
        print(f"  ✅ Đã có sẵn ảnh render từ trước.")

    # Copy ảnh render vào Submission với định dạng JPEG để tối ưu dung lượng (< 350MB)
    submission_dest = os.path.join(SUBMISSION_DIR, scene_name)
    if os.path.exists(render_path):
        os.makedirs(submission_dest, exist_ok=True)
        for img_path in glob.glob(os.path.join(render_path, "*.*")):
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_name = os.path.basename(img_path)
                out_jpg_path = os.path.join(submission_dest, img_name)
                
                # Nếu ảnh đã là JPEG (render.py đã lưu ở Q=98), chỉ cần copy trực tiếp để tránh nén lossy 2 lần
                if img_path.lower().endswith(('.jpg', '.jpeg')):
                    shutil.copy(img_path, out_jpg_path)
                else:
                    # Nếu là PNG thì mới convert sang JPEG để tiết kiệm dung lượng
                    try:
                        base_name = os.path.splitext(img_name)[0]
                        out_jpg_path = os.path.join(submission_dest, f"{base_name}.jpg")
                        with Image.open(img_path) as img:
                            rgb_im = img.convert('RGB')
                            rgb_im.save(out_jpg_path, 'JPEG', quality=98)
                    except Exception as e:
                        print(f"Lỗi khi chuyển đổi {img_name}: {e}")
        print(f"  ✅ Đã đóng gói các ảnh test vào thư mục nộp bài.")

print("\n" + "=" * 60)
print("3. ĐÓNG GÓI SUBMISSION")
print("=" * 60)
# Nén thư mục submission thành zip
zip_path = os.path.join(os.path.dirname(SUBMISSION_DIR), "submission")
shutil.make_archive(zip_path, 'zip', SUBMISSION_DIR)

final_zip_size = os.path.getsize(f"{zip_path}.zip") / (1024 * 1024)
print(f"Tạo file {zip_path}.zip thành công!")
print(f"Dung lượng file nén: {final_zip_size:.2f} MB")
if final_zip_size > 350:
    print("⚠️ CẢNH BÁO: Dung lượng file nén vượt quá 350MB! Hãy kiểm tra lại số lượng ảnh.")
else:
    print("✅ Đảm bảo giới hạn dung lượng < 350MB.")
print("Hoàn thành pipeline xử lý 8 scenes Private Set 1.")

# Dừng monitor GPU
if 'monitor' in locals():
    monitor.stop_event.set()

