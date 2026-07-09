import argparse
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
import sys

def sigmoid(x):
    # Dùng np.clip để tránh overflow khi exp
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

def main():
    parser = argparse.ArgumentParser(description="Làm sạch rác (Floaters) cho 3DGS Point Cloud")
    parser.add_argument("--input", type=str, required=True, help="Đường dẫn tới file .ply gốc")
    parser.add_argument("--output", type=str, required=True, help="Đường dẫn lưu file .ply đã được làm sạch")
    parser.add_argument("--max_scale", type=float, default=100.0, help="Kích thước tối đa cho phép (e^scale)")
    parser.add_argument("--min_opacity", type=float, default=0.05, help="Độ trong suốt tối thiểu cho phép (sigmoid(opacity))")
    parser.add_argument("--knn_n", type=int, default=15, help="Số lượng hàng xóm tối thiểu")
    parser.add_argument("--knn_r", type=float, default=0.5, help="Bán kính tìm kiếm hàng xóm (KNN)")
    
    args = parser.parse_args()
    
    print(f"[INFO] Đang tải point cloud từ {args.input}...")
    try:
        plydata = PlyData.read(args.input)
    except Exception as e:
        print(f"[ERROR] Không thể đọc file {args.input}: {e}")
        sys.exit(1)
        
    vertex_data = plydata['vertex'].data
    num_points_orig = len(vertex_data)
    print(f"[INFO] Tổng số Gaussians ban đầu: {num_points_orig}")
    
    # Tạo mask ban đầu (giữ lại tất cả)
    mask = np.ones(num_points_orig, dtype=bool)
    
    # ---------------------------------------------------------
    # Kỹ thuật 1: Scale Culling (Xóa Floaters khổng lồ)
    # ---------------------------------------------------------
    try:
        scale_0 = vertex_data['scale_0']
        scale_1 = vertex_data['scale_1']
        scale_2 = vertex_data['scale_2']
        
        # Scale được lưu dưới dạng log, nên scale thực tế là e^scale
        # Để tránh overflow, clip scale_raw trước
        scale_raw = np.vstack((scale_0, scale_1, scale_2)).T
        scale_raw = np.clip(scale_raw, -50, 50)
        actual_scales = np.exp(scale_raw)
        
        max_scales = np.max(actual_scales, axis=1)
        scale_mask = max_scales <= args.max_scale
        mask = mask & scale_mask
        print(f"[1/3] Scale Culling: Đã xóa {np.sum(~scale_mask)} Gaussians do kích thước vượt {args.max_scale}")
    except ValueError:
        print("[1/3] Scale Culling: Không tìm thấy thuộc tính scale_0/1/2, bỏ qua.")
        
    # ---------------------------------------------------------
    # Kỹ thuật 2: Opacity Culling (Xóa mây lờ mờ)
    # ---------------------------------------------------------
    try:
        opacity_raw = vertex_data['opacity']
        opacity_actual = sigmoid(opacity_raw)
        opacity_mask = opacity_actual >= args.min_opacity
        mask = mask & opacity_mask
        print(f"[2/3] Opacity Culling: Đã xóa {np.sum(~opacity_mask)} Gaussians do độ mờ < {args.min_opacity}")
    except ValueError:
        print("[2/3] Opacity Culling: Không tìm thấy thuộc tính opacity, bỏ qua.")
        
    # Lọc mask hiện tại trước khi vào KNN để tính toán nhanh hơn
    filtered_vertices = vertex_data[mask]
    print(f"[INFO] Số lượng Gaussians sau Scale & Opacity: {len(filtered_vertices)}")
    
    # ---------------------------------------------------------
    # Kỹ thuật 3: SOR / KNN (Bắn tỉa mây đơn độc)
    # ---------------------------------------------------------
    if len(filtered_vertices) > 0 and args.knn_n > 0 and args.knn_r > 0:
        x = filtered_vertices['x']
        y = filtered_vertices['y']
        z = filtered_vertices['z']
        pts = np.vstack((x, y, z)).T
        
        print(f"[3/3] KNN Culling: Đang xây dựng cây KDTree cho {len(pts)} điểm...")
        tree = cKDTree(pts)
        
        print(f"[3/3] KNN Culling: Đang tính khoảng cách đến {args.knn_n} hàng xóm (chạy đa luồng CPU)...")
        # Thay vì query_ball_point (rất chậm và tốn RAM với dữ liệu lớn),
        # ta tìm khoảng cách tới hàng xóm thứ knn_n. Nếu khoảng cách này <= knn_r
        # nghĩa là chắc chắn có ít nhất knn_n hàng xóm trong bán kính knn_r.
        distances, _ = tree.query(pts, k=args.knn_n, workers=-1)
        
        # distances[:, -1] là khoảng cách tới hàng xóm xa nhất trong K hàng xóm
        knn_mask = distances[:, -1] <= args.knn_r
        
        filtered_vertices = filtered_vertices[knn_mask]
        print(f"[3/3] KNN Culling: Đã xóa {np.sum(~knn_mask)} Gaussians mồ côi (khoảng cách tới hàng xóm thứ {args.knn_n} > {args.knn_r})")
    else:
        print("[3/3] KNN Culling: Bỏ qua do tham số không hợp lệ hoặc không còn Gaussians.")
        
    # ---------------------------------------------------------
    # Lưu file .ply mới
    # ---------------------------------------------------------
    num_points_final = len(filtered_vertices)
    print(f"[INFO] Tổng số Gaussians cuối cùng: {num_points_final} (còn lại {(num_points_final/num_points_orig)*100:.2f}%)")
    
    if num_points_final > 0:
        print(f"[INFO] Đang lưu file .ply sạch vào {args.output}...")
        el = PlyElement.describe(filtered_vertices, 'vertex')
        PlyData([el], text=False).write(args.output)
        print("[INFO] Hoàn thành xuất sắc!")
    else:
        print("[WARNING] Không còn Gaussian nào sau khi lọc! Không lưu file để tránh lỗi hệ thống.")

if __name__ == "__main__":
    main()
