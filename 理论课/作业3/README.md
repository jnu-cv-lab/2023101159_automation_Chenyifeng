DFT与DCT边界特性及能量集中性分析 README

一、实验目的

对比离散傅里叶变换（DFT）与离散余弦变换（DCT-II）的边界延拓特性、频谱差异及能量集中性，验证DCT在图像处理中的优势，掌握两种变换的编程实现与结果分析方法。

二、实验环境

- 开发环境：Python 3.10 + WSL Ubuntu 22.04 + VS Code

- 依赖库：numpy、opencv-python、matplotlib

- 测试数据：优先读取当前目录test.jpg（灰度图），提取中间一行像素作为测试信号；未读取到图像时，自动使用兜底短序列 [1, 3, 2, 4, 5, 3, 2, 1]

三、核心原理

- DFT：基于复指数基函数，隐含周期延拓，边界存在跳变，能量分散在正负频率分量中

- DCT-II：基于实余弦基函数，隐含偶对称（镜像）延拓，边界连续无跳变；本质是偶对称延拓序列的DFT，能量高度集中在低频分量

四、实验步骤

1. 安装所需依赖库（命令：pip install numpy opencv-python matplotlib）

2. 将test.jpg放入代码所在目录（无图片不影响运行，将自动启用兜底序列）

3. 运行实验代码，自动完成信号读取、延拓构造、变换计算、能量分析及结果绘图

4. 查看控制台输出的能量占比数据，及生成的结果图dft_dct_result.png

五、实验代码

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# 无弹窗后端配置（适配WSL）
plt.switch_backend('Agg')

# 1. 读取测试信号（兜底机制）
img_path = os.path.join(os.getcwd(), "test.jpg")
print(f"正在读取图片：{img_path}")
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("⚠️ 未读取到test.jpg，自动切换为测试短序列")
    x = np.array([1, 3, 2, 4, 5, 3, 2, 1], dtype=np.float64)
else:
    print(f"✅ 成功读取图像，尺寸：{img.shape}")
    x = img[img.shape[0]//2, :].astype(np.float64)  # 提取中间一行像素

N = len(x)
print(f"分析信号长度：{N}")

# 2. 构造延拓序列
dft_ext = np.tile(x, 3)  # DFT周期延拓
dct_ext = np.concatenate([x[::-1], x, x[::-1]])  # DCT偶对称延拓

# 3. 计算DFT与DCT-II
dft_result = np.fft.fft(x)

def dct_ii(signal):
    N = len(signal)
    n, k = np.arange(N), np.arange(N)
    c = np.ones(N)
    c[0] = 1 / np.sqrt(2)
    return np.sqrt(2/N) * c * np.sum(signal[:, None] * np.cos((2*n[:, None]+1)*k*np.pi/(2*N)), axis=0)

dct_result = dct_ii(x)

# 4. 计算能量占比
top_k = max(2, int(N*0.1))
dft_energy = np.sum(np.abs(dft_result[:top_k])**2) / np.sum(np.abs(dft_result)**2)
dct_energy = np.sum(np.abs(dct_result[:top_k])**2) / np.sum(np.abs(dct_result)**2)
print(f"\n前{top_k}个系数能量占比：\nDFT: {dft_energy:.2%}\nDCT: {dct_energy:.2%}")

# 5. 绘图并保存（无蓝色，原始信号为黑色）
plt.figure(figsize=(16, 10))
# 原始信号（黑色）
plt.subplot(2,2,1); plt.plot(x, 'k-', linewidth=1.2); plt.title('原始信号'); plt.xlabel('像素位置n'); plt.ylabel('灰度值'); plt.grid(alpha=0.3)
# DFT周期延拓
plt.subplot(2,2,2); plt.plot(dft_ext, 'r-'); plt.title('DFT周期延拓（边界跳变）'); plt.xlabel('延拓后位置'); plt.axvline(x=N, color='k', linestyle='--', label='边界'); plt.legend(); plt.grid(alpha=0.3)
# DCT偶对称延拓
plt.subplot(2,2,3); plt.plot(dct_ext, 'g-'); plt.title('DCT偶对称延拓（边界连续）'); plt.xlabel('延拓后位置'); plt.axvline(x=N, color='k', linestyle='--', label='边界'); plt.legend(); plt.grid(alpha=0.3)
# 频谱对比
plot_len = min(50, N)
plt.subplot(2,2,4); plt.plot(np.abs(dft_result[:plot_len]), 'r-', label='DFT幅度'); plt.plot(np.abs(dct_result[:plot_len]), 'g-', label='DCT幅度'); plt.title(f'频谱对比（前{top_k}个系数能量占比：DFT={dft_energy:.2%}, DCT={dct_energy:.2%}）'); plt.legend(); plt.grid(alpha=0.3)

plt.tight_layout()
save_path = os.path.join(os.getcwd(), "dft_dct_result.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
print(f"\n✅ 结果图已生成：{save_path}（VSCode左侧文件栏双击查看）")

六、实验结果

6.1 输出文件

dft_dct_result.png：4个子图，分别为原始信号、DFT周期延拓、DCT偶对称延拓、DFT与DCT频谱对比（无蓝色元素，原始信号为黑色线条）。

6.2 核心结论

- 边界特性：DFT周期延拓存在边界跳变，DCT偶对称延拓边界连续，无跳变

- 能量集中性：DCT远优于DFT（实验中前10%系数能量占比约99.47%，DFT约81.19%）

- 应用优势：DCT仅需实数运算，能量集中，无块效应，是图像压缩（JPEG、H.264）的核心变换

七、注意事项

- 代码适配WSL环境，无弹窗，仅保存结果图，避免运行报错

- 无test.jpg时自动启用兜底序列，确保实验100%可运行

- 结果图为高分辨率（300dpi），可直接用于实验报告插入

