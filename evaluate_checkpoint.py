import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import shutil
import torch
import torchvision
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

# Ensure script can import local BTS-Digital-Twin modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene import Scene
from gaussian_renderer import GaussianModel, render
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from arguments import ModelParams, PipelineParams, OptimizationParams

def find_scene_path(base_dir, scene_name):
    """Tìm thư mục chứa dữ liệu scene (không phân biệt hoa/thường)"""
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            if d.lower() == scene_name.lower():
                return os.path.join(root, d)
    return None

def main():
    parser = ArgumentParser(description="Đánh giá mô hình từ file checkpoint theo metrics chuẩn (LPIPS, SSIM, PSNR)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn đến file checkpoint (.pth)")
    parser.add_argument("--data_root", type=str, default=r"D:\ai_race_2026\data\phase1", help="Thư mục gốc chứa các dataset (ví dụ phase1/public_set)")
    parser.add_argument("--psnr_max", type=float, default=35.0, help="Ngưỡng PSNR_max để chuẩn hoá (PSNR_norm)")
    parser.add_argument("--save_renders_dir", type=str, default=None, help="Thư mục để lưu ảnh sinh ra (phục vụ tạo submission)")
    
    # Kế thừa các argument mặc định của hệ thống
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    args, _ = parser.parse_known_args()
    
    # 1. Trích xuất tên scene từ tên file checkpoint (ví dụ: chkpnt30000_hcm0181.pth -> hcm0181)
    ckpt_name = os.path.basename(args.checkpoint)
    scene_name = ckpt_name.replace("chkpnt", "").replace(".pth", "").split("_")[-1]
    print(f"[*] Tên scene được nhận diện: {scene_name}")
    
    # 2. Tìm thư mục dữ liệu gốc cho scene
    source_path = find_scene_path(args.data_root, scene_name)
    if not source_path:
        print(f"[!] Không tìm thấy thư mục dữ liệu cho scene '{scene_name}' trong {args.data_root}")
        sys.exit(1)
        
    print(f"[*] Đường dẫn dữ liệu (source_path): {source_path}")
    
    test_images_dir = os.path.join(source_path, "test", "images")
    if not os.path.exists(test_images_dir):
        print(f"[!] Không tìm thấy thư mục ảnh ground-truth test: {test_images_dir}. Không thể đánh giá.")
        sys.exit(1)

    # 3. Khởi tạo cấu hình và model
    # QUAN TRỌNG: lp.extract() trả về ModelParams, KHÔNG phải OptimizationParams
    # Đặt tên rõ ràng để tránh nhầm lẫn khi truyền vào gaussians.restore()
    model_cfg = lp.extract(args)
    model_cfg.source_path = source_path
    model_cfg.model_path = "./eval_output_temp"
    model_cfg.eval = True
    model_cfg.data_device = "cpu"  # Lưu ảnh gốc vào CPU RAM thay vì VRAM để tránh OOM
    os.makedirs(model_cfg.model_path, exist_ok=True)

    pipe = pp.extract(args)
    
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam
        SPARSE_ADAM_AVAILABLE = True
    except Exception:
        SPARSE_ADAM_AVAILABLE = False

    optimizer_type = getattr(model_cfg, "optimizer_type", "default")
    gaussians = GaussianModel(model_cfg.sh_degree, optimizer_type)

    print("[*] Đang khởi tạo Scene từ COLMAP data...")
    scene = Scene(model_cfg, gaussians, load_iteration=None, shuffle=False)

    print(f"[*] Đang nạp checkpoint: {args.checkpoint}")
    try:
        # weights_only=False: bắt buộc cho checkpoint chứa numpy scalar (optimizer states)
        # Tương thích với PyTorch >= 2.6 (default đã đổi từ False → True)
        (ckpt_model_params, first_iter) = torch.load(args.checkpoint, weights_only=False)
        # PHẢI truyền OptimizationParams vào restore(), không phải ModelParams
        optim_args = op.extract(args)
        gaussians.restore(ckpt_model_params, optim_args)
    except Exception as e:
        print(f"[!] Lỗi khi nạp checkpoint: {e}")
        sys.exit(1)
        
    # 4. Thiết lập môi trường đánh giá
    bg_color = [1, 1, 1] if model_cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_cameras = scene.getTestCameras()
    if not test_cameras:
        print("[!] Không tìm thấy test camera poses nào (test_poses.csv).")
        sys.exit(1)
        
    ssims = []
    psnrs = []
    lpipss = []
    
    print(f"[*] Bắt đầu render và tính toán metrics cho {len(test_cameras)} góc nhìn test...")
    
    with torch.no_grad():
        for view in tqdm(test_cameras, desc="Evaluation"):
            # Đọc ảnh ground-truth test
            gt_path = os.path.join(test_images_dir, view.image_name)
            if not os.path.exists(gt_path):
                print(f"  [!] GT image missing: {gt_path}, bỏ qua.")
                continue
                
            gt_img = Image.open(gt_path).convert("RGB")
            gt_tensor = tf.to_tensor(gt_img).unsqueeze(0).cuda() # (1, 3, H, W)
            
            # Render (SPARSE_ADAM_AVAILABLE quyết định có dùng separate_sh hay không)
            render_pkg = render(
                view, gaussians, pipe, background,
                use_trained_exp=getattr(model_cfg, "train_test_exp", False),
                separate_sh=SPARSE_ADAM_AVAILABLE
            )
            render_tensor = render_pkg["render"].unsqueeze(0).clamp(0.0, 1.0)  # (1, 3, H, W)

            # Resize GT nếu kích thước lệch (đảm bảo đồng nhất để tính metrics)
            if render_tensor.shape[2:] != gt_tensor.shape[2:]:
                gt_tensor = tf.resize(gt_tensor, [render_tensor.shape[2], render_tensor.shape[3]])

            # Lưu ảnh nếu có yêu cầu
            if args.save_renders_dir is not None:
                save_dir = os.path.join(args.save_renders_dir, scene_name)
                os.makedirs(save_dir, exist_ok=True)
                # torchvision.utils.save_image mong đợi tensor shape (C, H, W)
                torchvision.utils.save_image(render_tensor[0], os.path.join(save_dir, view.image_name))

            # Tính metrics — cả 3 metrics đều chạy trong torch.no_grad() block
            val_ssim = ssim(render_tensor, gt_tensor).item()
            val_psnr = psnr(render_tensor, gt_tensor).item()
            val_lpips = lpips(render_tensor, gt_tensor, net_type='vgg').item()

            ssims.append(val_ssim)
            psnrs.append(val_psnr)
            lpipss.append(val_lpips)

            # Giải phóng VRAM sau mỗi ảnh để tránh OOM
            del render_tensor, gt_tensor, render_pkg
            torch.cuda.empty_cache()
            
    if not ssims:
        print("[!] Không tính toán được kết quả cho bất kỳ ảnh nào.")
        sys.exit(1)
        
    # Dùng sum/len thay vì numpy để tránh thêm dependency khi không cần thiết
    n = len(ssims)
    mean_ssim  = sum(ssims)  / n
    mean_psnr  = sum(psnrs)  / n
    mean_lpips = sum(lpipss) / n
    
    # 5. Tính Score tổng hợp
    # Score = 0.4 × (1 − LPIPS) + 0.3 × SSIM + 0.3 × PSNRnorm
    psnr_norm = max(0.0, min(1.0, mean_psnr / args.psnr_max))
    score = 0.4 * (1.0 - mean_lpips) + 0.3 * mean_ssim + 0.3 * psnr_norm
    
    # Dọn thư mục temp (tránh tốn disk trên Kaggle)
    if os.path.exists(model_cfg.model_path) and model_cfg.model_path == "./eval_output_temp":
        try:
            shutil.rmtree(model_cfg.model_path)
        except Exception as e:
            print(f"[!] Không thể xoá thư mục temp {model_cfg.model_path}: {e}")

    print("\n" + "="*60)
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ: {scene_name.upper()}")
    print("="*60)
    print(f"  - SSIM       : {mean_ssim:.5f}")
    print(f"  - PSNR       : {mean_psnr:.5f} (Norm: {psnr_norm:.5f} với Max={args.psnr_max})")
    print(f"  - LPIPS      : {mean_lpips:.5f}")
    print("-" * 60)
    print(f"  🏆 SCORE TỔNG: {score:.5f}")
    print("="*60)

if __name__ == "__main__":
    main()
