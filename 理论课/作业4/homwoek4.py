import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

# ====================== 工具函数 ======================
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法加载图像：{path}")
    return img.astype(np.float32) / 255.0

def compute_local_freq_fft(img, block=32, overlap=0.5):
    h, w = img.shape
    step = max(1, int(block * (1 - overlap)))
    out_h = (h - block) // step + 1
    out_w = (w - block) // step + 1
    freq = np.zeros((out_h, out_w))

    hann_2d = np.hanning(block)[:, None] * np.hanning(block)
    f_vec = np.fft.fftfreq(block, 1)
    fx, fy = np.meshgrid(f_vec, f_vec)
    rad = np.sqrt(fx**2 + fy**2)

    for i in range(out_h):
        for j in range(out_w):
            y, x = i*step, j*step
            patch = img[y:y+block, x:x+block]

            f = np.fft.fft2(patch * hann_2d)
            f = np.fft.fftshift(f)
            power = np.abs(f)**2 + 1e-8

            idx = np.argsort(power.ravel())[::-1]
            p = power.ravel()[idx]
            r = rad.ravel()[idx]
            cum = np.cumsum(p) / np.sum(p)
            k = np.searchsorted(cum, 0.95)
            freq[i,j] = np.clip(r[k], 0, 0.5)

    return freq

def compute_gradient_features(img, block=32, overlap=0.5):
    h, w = img.shape
    step = max(1, int(block * (1 - overlap)))
    out_h = (h - block) // step + 1
    out_w = (w - block) // step + 1
    feat = np.zeros((out_h, out_w))

    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    mag = np.sqrt(gx**2 + gy**2)

    for i in range(out_h):
        for j in range(out_w):
            y, x = i*step, j*step
            feat[i,j] = np.mean(mag[y:y+block, x:x+block])

    return feat

# ====================== 纯 numpy 最优拟合 ======================
def fit_gradient_to_fft(fft_map, grad_feat):
    x = grad_feat.ravel()
    y = fft_map.ravel()
    
    k = np.cov(x, y)[0,1] / (np.var(x) + 1e-8)
    b = np.mean(y) - k * np.mean(x)
    pred = k * grad_feat + b
    return np.clip(pred, 0, 0.5)

# ====================== 主程序（完整可视化 + 全部子图） ======================
if __name__ == "__main__":
    img = load_image("test.jpg")

    # 计算频率图
    fft_freq = compute_local_freq_fft(img, block=32)
    grad_feat = compute_gradient_features(img, block=32)
    grad_freq = fit_gradient_to_fft(fft_freq, grad_feat)

    # 指标计算
    diff = np.abs(fft_freq - grad_freq)
    corr = np.corrcoef(fft_freq.ravel(), grad_freq.ravel())[0,1]
    mae = np.mean(diff)

    # ====================== 完整可视化（7 张子图全恢复） ======================
    plt.figure(figsize=(16,10))
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    # 1 原图
    plt.subplot(2,4,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image', fontweight='bold')
    plt.axis('off')

    # 2 FFT 频率图
    plt.subplot(2,4,2)
    plt.imshow(fft_freq, cmap='jet', vmin=0, vmax=0.5)
    plt.title('FFT 95% Energy Frequency', fontweight='bold')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

    # 3 梯度频率图（拟合后）
    plt.subplot(2,4,3)
    plt.imshow(grad_freq, cmap='jet', vmin=0, vmax=0.5)
    plt.title('Gradient-based Frequency (Fitted)', fontweight='bold')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

    # 4 误差图
    plt.subplot(2,4,4)
    plt.imshow(diff, cmap='hot', vmin=0, vmax=0.5)
    plt.title('Absolute Error', fontweight='bold')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

    # 5 散点图
    plt.subplot(2,4,5)
    plt.scatter(fft_freq.ravel(), grad_freq.ravel(), s=2, alpha=0.6)
    plt.plot([0,0.5],[0,0.5],'r--',lw=2)
    plt.xlabel('FFT')
    plt.ylabel('Gradient')
    plt.title(f'Correlation = {corr:.3f}', fontweight='bold')
    plt.grid(alpha=0.3)

    # 6 直方图
    plt.subplot(2,4,6)
    plt.hist(fft_freq.ravel(), bins=30, alpha=0.6, label='FFT')
    plt.hist(grad_freq.ravel(), bins=30, alpha=0.6, label='Gradient')
    plt.legend()
    plt.title('Frequency Distribution', fontweight='bold')

    # 7 统计表格
    plt.subplot(2,4,7)
    plt.axis('off')
    stats = f"""
    Statistics
    • FFT mean: {np.mean(fft_freq):.3f}
    • Gradient mean: {np.mean(grad_freq):.3f}
    • MAE: {mae:.3f}
    • Corr: {corr:.3f}
    """
    plt.text(0.5, 0.5, stats, fontsize=12, ha='center', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('frequency_result_final.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ====================== 最终输出 ======================
    print("======= ✅ 最终优化结果 =======")
    print(f"FFT 平均频率: {np.mean(fft_freq):.4f}")
    print(f"梯度平均频率: {np.mean(grad_freq):.4f}")
    print(f"平均绝对误差: {mae:.4f}")
    print(f"相关系数    : {corr:.4f}")