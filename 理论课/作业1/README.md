# YCbCr 图像下采样与插值重建实践
- 作者：陈亿锋
- 学号：2023101159
- 专业：自动化

## 项目概述
本作业围绕OpenCV库展开数字图像处理进阶实践，核心实现了图像的读取、YCbCr色彩空间转换、Cb/Cr通道下采样、双线性插值恢复、图像重建、PSNR质量评估、对比可视化及结果保存等功能。通过本次作业，掌握Python环境下OpenCV库对色彩空间的处理方法，理解YCbCr色彩空间「亮度-色度分离」的核心特性，熟悉下采样/插值对图像质量的影响及PSNR评估指标的应用，为图像压缩、画质优化等复杂处理奠定基础。

## 技术栈
编程语言：Python 3.8+（兼容3.7及以上版本）
核心库：
- OpenCV-Python（cv2）：图像读取、色彩空间转换、下采样插值、图像保存
- NumPy：图像数组操作、像素值计算（MSE/PSNR）
- Matplotlib：图像可视化展示、对比图生成与保存
- math：PSNR计算的数学运算（对数、开方）

## 文件清单
| 文件名 | 类型 | 功能说明 |
|--------|------|----------|
| text1.py | 源代码文件 | 主程序入口，包含所有图像处理逻辑：图片读取、YCbCr转换、下采样、插值、PSNR计算、对比图生成、结果保存 |
| test.jpg | 输入文件 | 原始图像测试，作为程序处理的输入源（建议使用jpg/png格式，避免特殊编码格式） |
| reconstructed_image.jpg | 输出文件 | 程序运行后自动生成的重建图像，保留原始图像尺寸，基于插值恢复的Cb/Cr通道重建 |
| comparison_image.jpg | 输出文件 | 程序运行后自动生成的对比图，包含原图与重建图，标注PSNR值，无坐标轴干扰 |
| README.md | 文档文件 | 项目完整说明，包含环境配置、运行步骤、功能解析、常见问题等 |

## 环境配置
### 1、环境要求
操作系统：Windows 11、Linux（Ubuntu 18.04+）、macOS 12+
Python版本：3.7及以上（推荐3.8-3.10，兼容性最佳）

### 2. 依赖安装
#### 方式一：pip直接安装（推荐）
打开终端/命令提示符，执行以下命令安装所有依赖：
```bash
# 安装 OpenCV-Python
pip install opencv-python
# 安装 NumPy（OpenCV 依赖，通常会自动安装）
pip install numpy
# 安装 Matplotlib（用于图像显示）
pip install matplotlib

# 批量安装（一行命令）
pip install opencv-python numpy matplotlib
```

#### 方式二：国内镜像源安装（解决下载慢/失败问题）
若直接安装超时或失败，使用清华镜像源加速：
```bash
pip install opencv-python numpy matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 方式三：固定版本安装（可选，保证环境一致性）
新建requirements.txt文件，写入以下内容：
```txt
opencv-python==4.8.1.78
numpy==1.24.3
matplotlib==3.7.2
```
执行安装命令：
```bash
pip install -r requirements.txt
```

## 运行步骤
### 1. 前期准备
将text1.py和test.jpg放在同一目录下，避免路径读取错误（若文件在src/目录下，需同步修改代码中图像读取路径为"src/test.jpg"）；
确认test.jpg文件未损坏，可通过系统图片查看器正常打开；
WSL环境需额外安装图形依赖（可选，解决Matplotlib显示问题）：
```bash
sudo apt install python3-tk libgtk2.0-0 -y
```

### 2、运行程序
#### 方式一：终端/命令行运行（推荐）
打开终端/命令提示符，切换到文件所在目录（示例：Windows系统）：
```bash
cd D:\homework\2023101159-automation-chenyifeng-homework
```
执行运行命令：
```bash
# Windows/Linux/macOS 通用
python text1.py
# 若系统存在多个 Python 版本，指定 Python3
python3 text1.py
```

#### 方式二：IDE运行（VS Code/PyCharm）
打开IDE并导入项目目录；
右键点击text1.py文件，选择「运行」/「Run Python File」；
等待程序执行，自动弹出图像显示窗口（可选）。

### 3. 运行结果
程序执行后会完成以下操作：
- 控制台输出：打印下采样倍数、PSNR值、对比图保存提示；
- 图像可视化：弹出包含「原始图像、重建图像」的显示窗口（无坐标轴干扰）；
- 文件生成：在项目目录下自动生成reconstructed_image.jpg（重建图）和comparison_image.jpg（对比图）。

## 核心功能解析
### 1. 图像读取与有效性验证
```python
import cv2
# 读取图像（OpenCV 默认 BGR 格式）
img = cv2.imread("test.jpg")
# 校验读取结果，避免文件不存在/损坏导致程序崩溃
if img is None:
    print("❌ 图片读取失败，请检查文件路径")
    exit()
```
核心函数：cv2.imread()返回图像的NumPy数组存储，None表示读取失败；
关键作用：提升程序健壮性，避免因文件路径/格式问题导致崩溃。

### 2. YCbCr色彩空间转换与通道分离
```python
# 获取图像尺寸（高, 宽）
h, w = img.shape[:2]
# BGR 转 YCbCr 色彩空间
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# 分离 Y（亮度）、Cr（红色差）、Cb（蓝色差）通道
Y, Cr, Cb = cv2.split(img_ycrcb)
```
转换规则：OpenCV通过cv2.COLOR_BGR2YCrCb实现BGR到YCbCr的映射；
通道特性：Y通道保留亮度信息，Cr/Cb通道保留色度信息，人眼对色度敏感度更低。

### 3. Cb/Cr通道下采样与插值恢复
```python
# 下采样倍数（2×2）
scale = 2
# 隔行隔列取像素，实现 Cr/Cb 通道下采样
Cb_down = Cb[::scale, ::scale]
Cr_down = Cr[::scale, ::scale]
# 双线性插值恢复至原始尺寸
interp_method = cv2.INTER_LINEAR
Cb_up = cv2.resize(Cb_down, (w, h), interpolation=interp_method)
Cr_up = cv2.resize(Cr_down, (w, h), interpolation=interp_method)
```
下采样原理：NumPy切片[::scale, ::scale]实现隔行隔列采样，降低色度通道分辨率；
插值规则：cv2.INTER_LINEAR（双线性插值）平衡重建质量与计算速度，是图像缩放的常用方法。

### 4. 图像重建与PSNR质量评估
```python
# 合并通道重建 YCbCr 图像
img_ycrcb_recon = cv2.merge((Y, Cr_up, Cb_up))
# YCbCr 转回 BGR 格式
img_rgb_recon = cv2.cvtColor(img_ycrcb_recon, cv2.COLOR_YCrCb2BGR)

# 定义 PSNR 计算函数（峰值信噪比，评估图像质量）
def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:  # MSE为0表示图像无失真
        return 100
    max_pixel = 255.0  # 像素值最大范围
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# 计算原图与重建图的 PSNR 值
psnr_value = calculate_psnr(img, img_rgb_recon)
print(f"📊 下采样倍数: {scale}×{scale} | PSNR: {psnr_value:.2f} dB")
```
重建逻辑：合并原始Y通道与插值恢复的Cr/Cb通道，保留亮度信息的同时还原色度；
PSNR解读：值越高图像质量越好，>30dB时人眼几乎无法区分原图与重建图。

### 5. Matplotlib生成对比可视化图
```python
import matplotlib.pyplot as plt
# BGR 转 RGB（适配 Matplotlib 显示规则）
img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_recon = cv2.cvtColor(img_rgb_recon, cv2.COLOR_BGR2RGB)

# 创建画布并显示对比图
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.imshow(img_original)
plt.title("Original Image", fontsize=14)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_recon)
plt.title(f"Reconstructed Image (PSNR: {psnr_value:.2f} dB)", fontsize=14)
plt.axis("off")

# 先保存再显示，避免保存的图片空白
plt.tight_layout()
plt.savefig("comparison_image.jpg", bbox_inches='tight', pad_inches=0.1)
plt.show(block=True)
plt.close()
```
核心问题：OpenCV读取为BGR格式，Matplotlib默认按RGB显示，直接显示会色彩失真；
关键细节：先执行plt.savefig()再执行plt.show()，避免显示窗口导致保存的对比图空白。

### 6. 重建图像保存
```python
# 保存重建后的BGR图像
cv2.imwrite("reconstructed_image.jpg", img_rgb_recon)
```
保存规则：cv2.imwrite()支持BGR格式直接保存，无需转换为RGB，与Matplotlib显示逻辑区分。
## 结果
- 原始图像尺寸：600 × 400
- 🔽 Cb/Cr通道下采样后尺寸：300 × 200

- 📈 下采样倍数：2×2
- 🔍 插值方法：1
- 💡 重建图像与原图的PSNR值：53.00 dB

- ✅ 结果已保存：
-    - original.jpg：原始图像
-    - reconstructed.jpg：下采样+插值重建后的图像
- 💡 提示：PSNR > 30 dB 时，人眼通常难以察觉图像质量差异

## 常见问题排查
| 问题现象 | 原因及解决方法 |
|----------|----------------|
| 程序提示「图片读取失败」 | 文件路径错误/名称大小写问题/格式不支持 → 确认文件同目录、名称一致、使用 jpg/png 格式 |
| 对比图保存空白 | 先执行plt.show()后执行plt.savefig() → 调整顺序：先保存再显示 |
| Matplotlib窗口不显示（WSL） | 缺失图形依赖 → 安装python3-tk，或注释plt.show()仅查看保存的对比图 |
| PSNR值为100 dB | 下采样倍数为1（无下采样）或图像无失真 → 检查scale变量是否为2 |
| 重建图色彩失真 | YCbCr通道合并顺序错误 → 确保合并顺序为(Y, Cr_up, Cb_up) |
| 安装库时报错 | 网络问题 → 使用国内镜像源；Python版本过低 → 升级到3.7+ |

## 作业总结
本次作业完成了基于OpenCV的YCbCr色彩空间图像处理实践，掌握了图像读取、色彩转换、下采样插值、PSNR评估、可视化展示等核心操作，理解了「亮度-色度分离」在图像压缩中的应用逻辑。
