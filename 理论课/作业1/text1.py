import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# ---------------------- 1. 读入图像并转换为YCbCr ----------------------
img = cv2.imread("test.jpg")
if img is None:
    print("❌ 图片读取失败，请检查文件路径")
    exit()

h, w = img.shape[:2]
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(img_ycrcb)

# ---------------------- 2. 下采样 + 插值恢复 ----------------------
scale = 2
Cb_down = Cb[::scale, ::scale]
Cr_down = Cr[::scale, ::scale]
interp_method = cv2.INTER_LINEAR
Cb_up = cv2.resize(Cb_down, (w, h), interpolation=interp_method)
Cr_up = cv2.resize(Cr_down, (w, h), interpolation=interp_method)

# ---------------------- 3. 重建 + PSNR计算 ----------------------
img_ycrcb_recon = cv2.merge((Y, Cr_up, Cb_up))
img_rgb_recon = cv2.cvtColor(img_ycrcb_recon, cv2.COLOR_YCrCb2BGR)

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

psnr_value = calculate_psnr(img, img_rgb_recon)
print(f"📊 下采样倍数: {scale}×{scale} | PSNR: {psnr_value:.2f} dB")

# ---------------------- 4. Matplotlib 生成对比图（关键：先保存再显示） ----------------------
# 转换为RGB（Matplotlib显示需要）
img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_recon = cv2.cvtColor(img_rgb_recon, cv2.COLOR_BGR2RGB)

# 创建画布
plt.figure(figsize=(15, 7))

# 显示原图
plt.subplot(1, 2, 1)
plt.imshow(img_original)
plt.title("Original Image", fontsize=14)
plt.axis("off")

# 显示重建图
plt.subplot(1, 2, 2)
plt.imshow(img_recon)
plt.title(f"Reconstructed Image (PSNR: {psnr_value:.2f} dB)", fontsize=14)
plt.axis("off")

# ✅ 先保存图片，再显示窗口（顺序不能变！）
plt.tight_layout()
plt.savefig("comparison_image.jpg", bbox_inches='tight', pad_inches=0.1)
print("✅ 对比图已保存为: comparison_image.jpg")

# 弹出窗口显示（可选，若不需要可注释掉）
plt.show(block=True)
plt.close()

# ---------------------- 5. 保存重建图 ----------------------
cv2.imwrite("reconstructed_image.jpg", img_rgb_recon)