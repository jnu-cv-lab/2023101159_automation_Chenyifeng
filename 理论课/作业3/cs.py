import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

plt.switch_backend('Agg')

# -------------------------- 第一步：读取图像--------------------------
# 直接读取当前文件夹里的test.jpg，绝对路径兜底
img_path = os.path.join(os.getcwd(), "test.jpg")
print(f"正在读取图片：{img_path}")
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 兜底：如果读不到图，自动切换为短序列，绝对不报错
if img is None:
    print("⚠️ 未读取到test.jpg，自动切换为测试短序列")
    x = np.array([1, 3, 2, 4, 5, 3, 2, 1], dtype=np.float64)
else:
    print(f"✅ 成功读取图像，尺寸：{img.shape}")
    # 提取图像中间一行像素，完全符合作业要求
    row_idx = img.shape[0] // 2
    x = img[row_idx, :].astype(np.float64)

N = len(x)
print(f"分析信号长度：{N}")

# -------------------------- 第二步：延拓方式对比 --------------------------
# DFT周期延拓（重复3次，模拟周期边界）
dft_ext = np.tile(x, 3)
# DCT偶对称延拓（左右镜像，模拟偶对称边界）
dct_ext_left = x[::-1]
dct_ext_right = x[::-1]
dct_ext = np.concatenate([dct_ext_left, x, dct_ext_right])

# -------------------------- 第三步：计算DFT和DCT --------------------------

dft_result = np.fft.fft(x)

# 手动实现DCT-II，完全符合作业定义，绝对不报错
def dct_ii(signal):
    N = len(signal)
    n = np.arange(N)
    k = np.arange(N)
    c = np.ones(N)
    c[0] = 1 / np.sqrt(2)
    X = np.sqrt(2 / N) * c * np.sum(
        signal[:, None] * np.cos((2 * n[:, None] + 1) * k * np.pi / (2 * N)),
        axis=0
    )
    return X

dct_result = dct_ii(x)

# -------------------------- 第四步：计算能量占比 --------------------------
top_k = max(2, int(N * 0.1))  # 至少取前2个系数
dft_energy = np.sum(np.abs(dft_result[:top_k])**2) / np.sum(np.abs(dft_result)**2)
dct_energy = np.sum(np.abs(dct_result[:top_k])**2) / np.sum(np.abs(dct_result)**2)
print(f"\n前{top_k}个系数能量占比：")
print(f"DFT: {dft_energy:.2%}")
print(f"DCT: {dct_energy:.2%}")

# -------------------------- 第五步：绘图并保存--------------------------
plt.figure(figsize=(16, 10))

# 子图1：原始信号
plt.subplot(2, 2, 1)
plt.plot(x, 'b-', linewidth=1.2)
plt.title('原始信号（灰度图中间一行像素）', fontsize=12)
plt.xlabel('像素位置 n', fontsize=10)
plt.ylabel('灰度值', fontsize=10)
plt.grid(alpha=0.3)

# 子图2：DFT周期延拓
plt.subplot(2, 2, 2)
plt.plot(dft_ext, 'r-', linewidth=1.2)
plt.title('DFT周期延拓（边界存在跳变）', fontsize=12)
plt.xlabel('延拓后位置', fontsize=10)
plt.ylabel('灰度值', fontsize=10)
plt.grid(alpha=0.3)
plt.axvline(x=N, color='k', linestyle='--', alpha=0.7, label='边界位置')
plt.axvline(x=2*N, color='k', linestyle='--', alpha=0.7)
plt.legend()

# 子图3：DCT偶对称延拓
plt.subplot(2, 2, 3)
plt.plot(dct_ext, 'g-', linewidth=1.2)
plt.title('DCT偶对称延拓（边界连续无跳变）', fontsize=12)
plt.xlabel('延拓后位置', fontsize=10)
plt.ylabel('灰度值', fontsize=10)
plt.grid(alpha=0.3)
plt.axvline(x=N, color='k', linestyle='--', alpha=0.7, label='边界位置')
plt.axvline(x=2*N, color='k', linestyle='--', alpha=0.7)
plt.legend()

# 子图4：频谱对比
plt.subplot(2, 2, 4)
# 只画前50个系数，避免长序列图太挤
plot_len = min(50, N)
plt.plot(np.abs(dft_result[:plot_len]), 'r-', linewidth=1.2, label='DFT幅度')
plt.plot(np.abs(dct_result[:plot_len]), 'g-', linewidth=1.2, label='DCT幅度')
plt.title(f'DFT/DCT频谱对比\n前{top_k}个系数能量占比：DFT={dft_energy:.2%}, DCT={dct_energy:.2%}', fontsize=12)
plt.xlabel('频率系数 k', fontsize=10)
plt.ylabel('幅度 |X[k]|', fontsize=10)
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
save_path = os.path.join(os.getcwd(), "dft_dct_result.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ 结果图已生成，路径：{save_path}")
print("直接在VSCode左侧文件栏双击打开即可查看！")