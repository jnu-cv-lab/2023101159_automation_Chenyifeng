import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# =========================
# 工具函数：兼容中文路径的图片读写
# =========================
def read_image_cn(path: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray | None:
    """兼容中文/特殊字符路径的图片读取"""
    if not path.exists():
        return None
    img_array = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(img_array, flags)


def save_image_cn(save_path: Path, img: np.ndarray) -> bool:
    """兼容中文/特殊字符路径的图片保存"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded_img = cv2.imencode(".jpg", img)
    if success:
        encoded_img.tofile(str(save_path))
    return success


# =========================
# 1. 配置参数集中管理
# =========================
# 路径配置（已修改为你的实际图片路径）
IMAGE_DIR = Path("/home/gevle/cv-course/原图")       # 你的16张原图所在文件夹
OUTPUT_DIR = Path("/home/gevle/cv-course/标定结果")  # 结果输出文件夹

# 棋盘格参数（请确认和你的实际棋盘格一致）
CHESSBOARD_SIZE = (9, 6)  # 内角点数量：列数 × 行数
SQUARE_SIZE = 25.0        # 方格实际边长，单位：mm；如果你的是30mm请改成30.0

# 角点检测参数
CORNER_SUBPIX_WINSIZE = (11, 11)
CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 标定参数：默认使用5参数畸变模型 [k1,k2,p1,p2,k3]
# 普通手机镜头可添加 cv2.CALIB_FIX_K3 固定k3=0，减少过拟合
CALIBRATION_FLAGS = 0

# 去畸变参数
# alpha=0: 裁剪黑边，保留有效像素；alpha=1: 保留全部像素，周围带黑边
UNDISTORT_ALPHA = 0


# =========================
# 2. 构造棋盘格三维世界坐标
# =========================
def generate_object_points(pattern_size, square_size):
    """生成棋盘格角点的世界坐标系三维坐标（Z=0平面）"""
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


# =========================
# 3. 批量检测角点
# =========================
def detect_chessboard_corners(image_paths, pattern_size, criteria, subpix_winsize):
    """批量检测棋盘格角点，返回三维点、二维点、成功图片路径、失败图片名、图像尺寸"""
    objpoints = []
    imgpoints = []
    success_paths = []
    failed_names = []
    image_size = None

    # 角点检测flags：自适应阈值+亮度归一化+快速预检，提升检测率与速度
    detect_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )

    objp_template = generate_object_points(pattern_size, SQUARE_SIZE)

    for idx, img_path in enumerate(image_paths):
        print(f"[{idx+1}/{len(image_paths)}] 处理图片：{img_path.name}")

        img = read_image_cn(img_path)
        if img is None:
            print("  × 图片读取失败，跳过")
            failed_names.append(img_path.name)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]

        # 校验所有图片分辨率一致（标定强制要求）
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            print(f"  × 分辨率不一致（期望{image_size}，实际{current_size}），跳过")
            failed_names.append(img_path.name)
            continue

        # 检测角点
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags=detect_flags)

        if ret:
            # 亚像素精度优化
            corners_subpix = cv2.cornerSubPix(
                gray, corners, subpix_winsize, (-1, -1), criteria
            )

            objpoints.append(objp_template)
            imgpoints.append(corners_subpix)
            success_paths.append(img_path)

            # 绘制并保存角点检测结果
            img_draw = img.copy()
            cv2.drawChessboardCorners(img_draw, pattern_size, corners_subpix, ret)
            save_path = OUTPUT_DIR / f"角点检测_{idx+1:02d}_{img_path.stem}.jpg"
            save_image_cn(save_path, img_draw)
            print("  √ 角点检测成功，已保存结果图")
        else:
            failed_names.append(img_path.name)
            print("  × 角点检测失败")

    return objpoints, imgpoints, success_paths, failed_names, image_size


# =========================
# 4. 执行相机标定
# =========================
def run_calibration(objpoints, imgpoints, image_size, flags=0):
    """执行相机标定，返回标定结果与误差统计"""
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags
    )

    # 手动计算每张图的平均重投影误差（L2范数逐点平均）
    per_image_errors = []
    total_error = 0.0
    for i in range(len(objpoints)):
        proj_points, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], proj_points, cv2.NORM_L2) / len(proj_points)
        per_image_errors.append(error)
        total_error += error
    mean_error = total_error / len(objpoints)

    return {
        "rms_error": ret,
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "per_image_errors": per_image_errors,
        "mean_error": mean_error
    }


# =========================
# 5. 去畸变与对比图生成
# =========================
def generate_undistort_result(img_path, camera_matrix, dist_coeffs, alpha=0):
    """生成单张图片的去畸变结果与对比图"""
    img = read_image_cn(img_path)
    h, w = img.shape[:2]

    # 优化相机矩阵，控制去畸变后的有效视野
    new_cam_mat, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
    )

    # 执行去畸变
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_cam_mat)

    # 裁剪有效区域（去除黑边）
    x, y, roi_w, roi_h = roi
    if roi_w > 0 and roi_h > 0:
        undistorted_crop = undistorted[y:y+roi_h, x:x+roi_w]
    else:
        undistorted_crop = undistorted

    # 保存结果图
    save_image_cn(OUTPUT_DIR / "去畸变原图.jpg", img)
    save_image_cn(OUTPUT_DIR / "去畸变结果.jpg", undistorted)
    save_image_cn(OUTPUT_DIR / "去畸变结果_裁剪版.jpg", undistorted_crop)

    # 生成并排对比图
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undist_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(undist_rgb)
    plt.title("Undistorted Image")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "原图vs去畸变对比图.png", dpi=300, bbox_inches="tight")
    plt.close()

    return undistorted, new_cam_mat, roi


# =========================
# 6. 结果保存与可视化
# =========================
def save_calibration_results(calib_result, success_paths, failed_names, image_size):
    """保存标定结果到文件，适配实验报告的数据需求"""
    # 保存npz格式的完整数据（可后续复用）
    np.savez(
        OUTPUT_DIR / "相机标定结果.npz",
        camera_matrix=calib_result["camera_matrix"],
        dist_coeffs=calib_result["dist_coeffs"],
        rvecs=calib_result["rvecs"],
        tvecs=calib_result["tvecs"],
        image_size=image_size,
        reprojection_mean_error=calib_result["mean_error"],
        reprojection_rms_error=calib_result["rms_error"]
    )

    # 生成格式化文本报告
    cm = calib_result["camera_matrix"]
    dist = calib_result["dist_coeffs"].flatten()
    cx, cy = cm[0, 2], cm[1, 2]
    center_x, center_y = image_size[0]/2, image_size[1]/2

    with open(OUTPUT_DIR / "标定结果报告.txt", "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("相机标定结果报告\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"图像分辨率：{image_size[0]} × {image_size[1]}\n")
        f.write(f"有效标定图片数：{len(success_paths)}\n")
        f.write(f"RMS 重投影误差：{calib_result['rms_error']:.4f} pixel\n")
        f.write(f"平均重投影误差：{calib_result['mean_error']:.4f} pixel\n\n")

        f.write("-" * 30 + "\n")
        f.write("相机内参矩阵 K\n")
        f.write("-" * 30 + "\n")
        f.write(f"fx = {cm[0,0]:.2f} pixel\n")
        f.write(f"fy = {cm[1,1]:.2f} pixel\n")
        f.write(f"cx = {cx:.2f} pixel (图像中心x = {center_x:.2f})\n")
        f.write(f"cy = {cy:.2f} pixel (图像中心y = {center_y:.2f})\n")
        f.write(f"主点偏移量：({abs(cx-center_x):.2f}, {abs(cy-center_y):.2f}) pixel\n\n")

        f.write("-" * 30 + "\n")
        f.write("畸变参数 D = [k1, k2, p1, p2, k3]\n")
        f.write("-" * 30 + "\n")
        f.write(f"k1 = {dist[0]:.6f}\n")
        f.write(f"k2 = {dist[1]:.6f}\n")
        f.write(f"p1 = {dist[2]:.6f}\n")
        f.write(f"p2 = {dist[3]:.6f}\n")
        f.write(f"k3 = {dist[4]:.6f}\n\n")

        f.write("-" * 30 + "\n")
        f.write("各图片重投影误差\n")
        f.write("-" * 30 + "\n")
        for i, err in enumerate(calib_result["per_image_errors"]):
            f.write(f"第 {i+1:2d} 张 ({success_paths[i].name}): {err:.4f} pixel\n")

        if failed_names:
            f.write("\n" + "-" * 30 + "\n")
            f.write("角点检测失败的图片\n")
            f.write("-" * 30 + "\n")
            for name in failed_names:
                f.write(f"- {name}\n")

    # 生成重投影误差柱状图，直观定位坏图
    plt.figure(figsize=(10, 5))
    x = range(1, len(calib_result["per_image_errors"])+1)
    plt.bar(x, calib_result["per_image_errors"], color="#4C72B0")
    plt.axhline(
        y=calib_result["mean_error"], 
        color="#C44E52", 
        linestyle="--", 
        label=f"平均误差: {calib_result['mean_error']:.4f}"
    )
    plt.xlabel("Image Index")
    plt.ylabel("Reprojection Error (pixel)")
    plt.title("Per-image Reprojection Error")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "重投影误差分布图.png", dpi=300)
    plt.close()


# =========================
# 主程序入口
# =========================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 获取所有图片路径
    image_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    image_paths = []
    for ext in image_exts:
        image_paths.extend(IMAGE_DIR.glob(ext))
    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"在 {IMAGE_DIR} 中未找到任何图片")
    print(f"共找到 {len(image_paths)} 张图片，开始角点检测...\n")

    # 2. 批量检测角点
    objpoints, imgpoints, success_paths, failed_names, image_size = detect_chessboard_corners(
        image_paths, CHESSBOARD_SIZE, CORNER_CRITERIA, CORNER_SUBPIX_WINSIZE
    )

    success_count = len(success_paths)
    print(f"\n{'='*40}")
    print(f"角点检测完成：成功 {success_count} 张，失败 {len(failed_names)} 张")
    print(f"{'='*40}\n")

    if success_count < 10:
        print("⚠️  警告：有效图片数量少于10张，标定结果可能不可靠，建议补充图片")
    if success_count < 3:
        raise RuntimeError("有效图片不足3张，无法进行标定")

    # 3. 执行标定
    print("开始相机标定...")
    calib_result = run_calibration(objpoints, imgpoints, image_size, CALIBRATION_FLAGS)
    print("标定完成\n")

    # 4. 结果合理性检查
    cm = calib_result["camera_matrix"]
    print("="*40)
    print("标定结果概览")
    print("="*40)
    print(f"RMS重投影误差：{calib_result['rms_error']:.4f} pixel")
    print(f"平均重投影误差：{calib_result['mean_error']:.4f} pixel")
    print(f"焦距 fx={cm[0,0]:.2f}, fy={cm[1,1]:.2f}，比值 fx/fy={cm[0,0]/cm[1,1]:.4f}")
    print(f"主点 ({cm[0,2]:.2f}, {cm[1,2]:.2f})，图像中心 ({image_size[0]/2:.2f}, {image_size[1]/2:.2f})")

    if calib_result["mean_error"] > 1.0:
        print("⚠️  警告：平均重投影误差大于1像素，标定精度偏低，建议检查图片质量")
    else:
        print("✅ 重投影误差在合理范围内")

    # 5. 生成去畸变结果
    print("\n生成去畸变对比图...")
    generate_undistort_result(
        success_paths[0], 
        calib_result["camera_matrix"], 
        calib_result["dist_coeffs"], 
        UNDISTORT_ALPHA
    )

    # 6. 保存所有结果
    save_calibration_results(calib_result, success_paths, failed_names, image_size)

    print(f"\n所有结果已保存至：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()