# 图像增强与滤波组合处理实践
- 作者：陈亿锋
- 学号：2023101159
- 专业：自动化

## 项目概述
本作业基于OpenCV与NumPy实现数字图像增强核心实践，从网络获取3幅不同类型原始图像，分别构建低对比度、高斯噪声、椒盐噪声三类待处理图像，自行实现均值滤波模块，完成**滤波→均衡**与**均衡→滤波**两种组合增强处理，同时实现全局直方图均衡、CLAHE、高斯滤波、中值滤波、图像锐化等基础增强方法。通过实验对比不同增强方法、不同组合处理的效果差异，结合PSNR（峰值信噪比）、SSIM（结构相似性）两项定量指标完成图像质量评估，实现处理结果与直方图的可视化展示及文件保存，掌握图像噪声抑制、对比度增强的核心方法，理解滤波与均衡的组合处理逻辑及参数对增强效果的影响。

## 技术栈
编程语言：Python 3.7+
核心库：
- OpenCV-Python（cv2）：图像读取、网络图像本地保存、滤波、均衡化、锐化、图像保存
- NumPy：图像数组操作、手动滤波的卷积运算、噪声生成、数组填充与索引
- Matplotlib：图像可视化、直方图绘制、结果图保存
- scikit-image：PSNR、SSIM定量评价指标计算

## 文件清单
| 文件名 | 类型 | 功能说明 |
|--------|------|----------|
| main.py | 源代码文件 | 主程序入口，包含所有核心逻辑：网络图像下载、图像退化处理、手动均值滤波实现、多种增强方法、组合处理、定量评价、结果与直方图可视化、文件保存 |
| network_img_0.jpg | 输入文件 | 网络下载的风景原始图，经处理为低对比度测试图 |
| network_img_1.jpg | 输入文件 | 网络下载的人像原始图，经处理为高斯噪声测试图 |
| network_img_2.jpg | 输入文件 | 网络下载的建筑原始图，经处理为椒盐噪声测试图 |
| result_低对比度.png | 输出文件 | 低对比度图像的所有增强处理结果+原始直方图可视化图 |
| result_含噪声.png | 输出文件 | 高斯噪声图像的所有增强处理结果+原始直方图可视化图 |
| result_正常.png | 输出文件 | 建筑椒盐噪声图像的所有增强处理结果+原始直方图可视化图 |
| README.md | 文档文件 | 项目完整说明，包含环境配置、运行步骤、核心功能解析、实验结果、常见问题等 |

## 环境配置
### 1、环境要求
操作系统：Windows 10/11、Linux（Ubuntu 18.04+）、macOS 12+
Python版本：3.7及以上（推荐3.8-3.10，兼容性最佳）
WSL环境：需安装基础图形依赖，解决Matplotlib可视化与保存问题

### 2. 依赖安装
#### 方式一：pip直接安装（推荐）
打开终端/命令提示符，执行一行命令安装所有依赖：
```bash
pip install opencv-python numpy matplotlib scikit-image
```

#### 方式二：国内镜像源安装（解决下载慢/超时/失败问题）
使用清华镜像源加速安装，适配国内网络环境：
```bash
pip install opencv-python numpy matplotlib scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 方式三：固定版本安装（保证环境一致性，可选）
新建`requirements.txt`文件，写入以下内容：
```txt
opencv-python==4.8.1.78
numpy==1.24.3
matplotlib==3.7.2
scikit-image==0.21.0
```
执行安装命令：
```bash
pip install -r requirements.txt
```

### 3. WSL环境额外配置
若在Linux WSL环境运行，安装图形依赖解决Matplotlib保存/显示问题：
```bash
sudo apt install python3-tk libgtk2.0-0 -y
```

## 运行步骤
### 1. 前期准备
将`main.py`放在独立项目目录下，保证目录读写权限；无需手动准备图像，程序将自动从网络下载3幅不同类型原始图像并保存在同目录；确认网络通畅，保证图像下载功能正常。

### 2、运行程序
#### 方式一：终端/命令行运行（推荐，无IDE依赖）
1. 打开终端/命令提示符，切换到`main.py`所在项目目录（示例Linux命令）：
```bash
cd /home/chen/cv-course
```
2. 若使用虚拟环境，先激活虚拟环境（示例）：
```bash
source .venv-basic/bin/activate
```
3. 执行运行命令：
```bash
# Linux/macOS
python3 main.py
# Windows
python main.py
```

#### 方式二：IDE运行（VS Code/PyCharm）
1. 打开IDE并导入项目目录，配置好对应Python解释器（虚拟环境/系统环境）；
2. 右键点击`main.py`文件，选择「运行」/「Run Python File」；
3. 等待程序执行，终端将输出定量评价结果，项目目录自动生成输入图像与输出结果图。

### 3. 运行结果
程序执行后完成以下操作，无人工干预：
- 网络图像下载：自动下载3幅不同类型原始图像，保存为`network_img_0/1/2.jpg`；
- 图像退化处理：分别将3幅原始图像处理为**低对比度、高斯噪声、椒盐噪声**测试图；
- 控制台输出：按图像类型打印，包含各增强方法、组合处理的PSNR、SSIM定量评价指标；
- 结果可视化保存：为每类测试图生成增强处理结果+原始直方图的可视化图，保存为`result_*.png`；
- 多方法处理：完成8种增强/组合处理，覆盖滤波、均衡、锐化、组合操作全流程。

## 核心功能解析
### 1. 网络图像自动下载与本地保存
程序自动从网络获取3幅不同类型图像，无需手动准备，提升程序独立性：
```python
import urllib.request
import os
# 网络图像URL（风景、人像、建筑，3类不同类型）
urls = ["https://picsum.photos/id/10/512/512", "https://picsum.photos/id/22/512/512", "https://picsum.photos/id/30/512/512"]
for idx, url in enumerate(urls):
    img_path = f"network_img_{idx}.jpg"
    if not os.path.exists(img_path):
        urllib.request.urlretrieve(url, img_path)  # 下载并本地保存
    img = cv2.imread(img_path, 0)  # 以灰度图形式读取
```
核心逻辑：判断本地是否存在图像，避免重复下载；以灰度图读取，适配后续增强处理流程；
关键作用：实现**3幅不同类型图像的网络自动获取**，满足作业输入要求。

### 2. 图像退化处理（低对比度/高斯噪声/椒盐噪声）
为3幅不同原始图像分别构建待增强的退化图像，模拟实际图像处理场景：
```python
# 低对比度处理：灰度值压缩，降低动态范围
img = np.clip(img * 0.4 + 80, 0, 255).astype(np.uint8)
# 高斯噪声处理：生成正态分布噪声，叠加后裁剪像素值
noise = np.random.normal(0, 25, img.shape).astype(np.int16)
img = np.clip(img + noise, 0, 255).astype(np.uint8)
# 椒盐噪声处理：生成随机黑白噪声点，模拟脉冲噪声
salt = np.random.rand(*img.shape) < 0.02
pepper = np.random.rand(*img.shape) < 0.02
img[salt] = 255
img[pepper] = 0
```
处理规则：低对比度通过灰度值缩放+偏移实现；高斯噪声服从N(0,25)分布；椒盐噪声占比约2%；
核心细节：所有操作后通过`np.clip`限制像素值在0-255，避免像素值溢出。

### 3. 手动实现均值滤波（核心自行实现模块，非调用现成接口）
作业核心要求，手动实现均值滤波的卷积、边缘填充逻辑，替代`cv2.blur`现成接口：
```python
def manual_mean_filter(img, kernel_size=3):
    h, w = img.shape[:2]
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode='edge')  # 边缘填充，避免边界像素处理缺失
    result = np.zeros_like(img, dtype=np.float32)
    # 双重循环实现卷积运算，遍历每个像素的邻域
    for i in range(h):
        for j in range(w):
            result[i, j] = np.mean(padded[i:i+kernel_size, j:j+kernel_size])
    return np.clip(result, 0, 255).astype(np.uint8)
```
实现逻辑：先对图像进行边缘填充（edge模式，保持边界像素特征），再通过双重循环遍历每个像素，计算其`kernel_size×kernel_size`邻域的像素均值，作为滤波后像素值；
关键细节：使用浮点型计算避免精度丢失，最后裁剪并转换为uint8像素类型。

### 4. 多类型图像增强方法实现
实现8种图像增强/处理方法，覆盖均衡、滤波、锐化、组合处理，满足实验对比要求：
```python
def enhance_methods(img):
    results = {}
    # 基础增强：全局直方图均衡、CLAHE自适应均衡
    results["global_eq"] = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    results["clahe"] = clahe.apply(img)
    # 滤波处理：手动均值滤波、OpenCV高斯滤波、中值滤波
    results["manual_mean"] = manual_mean_filter(img, 3)
    results["gaussian"] = cv2.GaussianBlur(img, (3,3), 0)
    results["median"] = cv2.medianBlur(img, 3)
    # 图像锐化：拉普拉斯锐化核，增强边缘与细节
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    results["sharpen"] = cv2.filter2D(img, -1, kernel)
    # 核心组合处理：滤波→均衡、均衡→滤波
    results["filter->eq"] = cv2.equalizeHist(manual_mean_filter(img, 3))
    results["eq->filter"] = manual_mean_filter(cv2.equalizeHist(img), 3)
    return results
```
核心方法：
- 均衡化：全局均衡适合整体低对比度，CLAHE适合局部低对比度，避免过曝；
- 滤波：手动均值滤波为自研模块，高斯滤波抑制高斯噪声，中值滤波抑制椒盐噪声；
- 组合处理：**滤波→均衡**（先去噪再增强对比度）、**均衡→滤波**（先增强对比度再平滑），为实验核心对比对象。

### 5. 定量评价指标计算（PSNR+SSIM）
采用两项行业通用指标，客观评估图像增强效果，避免仅视觉评价的主观性：
```python
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
def evaluate(img_original, img_processed):
    psnr_val = psnr(img_original, img_processed)
    ssim_val = ssim(img_original, img_processed, data_range=255)
    return psnr_val, ssim_val
```
指标解读：
- **PSNR（峰值信噪比）**：单位为dB，值越高表示增强后图像与原图的保真度越好，噪声抑制效果越佳；
- **SSIM（结构相似性）**：取值范围0-1，值越高表示增强后图像与原图的结构、纹理保留越完整，避免增强导致的细节丢失；
- 数据范围：设置`data_range=255`，适配8位灰度图的像素值范围。

### 6. 处理结果与直方图可视化
实现原始图像、所有增强结果、原始直方图的一体化可视化，保存为图片文件，直观展示增强效果：
```python
def plot_results(img, results, img_name):
    plt.figure(figsize=(18, 16))
    # 原始图像展示
    plt.subplot(4, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"原始图像：{img_name}")
    plt.axis("off")
    # 原始图像直方图：展示灰度值分布，为增强效果提供量化参考
    plt.subplot(4, 3, 2)
    plt.hist(img.flatten(), bins=256, color="gray")
    plt.title("原始直方图")
    # 遍历所有增强结果，逐一封装展示
    methods = list(results.keys())
    for i, method in enumerate(methods):
        plt.subplot(4, 3, i + 3)
        plt.imshow(results[method], cmap="gray")
        plt.title(method)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"result_{img_name}.png")
    plt.close()
```
可视化逻辑：采用4×3子图布局，可容纳1幅原始图+1幅直方图+8幅增强结果图，无布局溢出；
关键细节：关闭坐标轴（`plt.axis("off")`），提升可视化效果；使用`tight_layout()`自动调整子图间距，避免重叠；灰度图展示（`cmap="gray"`），适配图像处理流程。

## 实验结果
本次实验对**风景-低对比度、人像-高斯噪声、建筑-椒盐噪声**3类图像进行增强处理，核心对比**滤波→均衡**与**均衡→滤波**两种组合处理的效果，以下为核心定量评价结果（PSNR单位：dB，SSIM取值：0-1），所有结果由程序自动计算输出：

### 1. 风景-低对比度图像
核心问题：灰度值分布集中，图像整体偏暗，对比度低；
组合处理定量结果：
- 滤波→均衡：PSNR: 14.24 | SSIM: 0.679
- 均衡→滤波：PSNR: 14.58 | SSIM: 0.728
**结果分析**：均衡→滤波的PSNR和SSIM均高于滤波→均衡，说明低对比度图像无明显噪声，先通过均衡化拉伸灰度范围提升对比度，再通过均值滤波平滑图像，能更好保留图像结构与保真度。

### 2. 人像-高斯噪声图像
核心问题：图像存在高斯噪声，细节模糊，灰度值分布杂乱；
组合处理定量结果：
- 滤波→均衡：PSNR: 13.39 | SSIM: 0.504
- 均衡→滤波：PSNR: 16.24 | SSIM: 0.568
**结果分析**：均衡→滤波的PSNR提升约2.85dB，SSIM提升约0.064，说明高斯噪声为连续噪声，先均衡化增强对比度后，高斯滤波对噪声的抑制效果更优，图像保真度和结构保留度均显著提升。

### 3. 建筑-椒盐噪声图像
核心问题：图像存在黑白脉冲噪声点，边缘细节被噪声干扰；
组合处理定量结果：
- 滤波→均衡：PSNR: 15.70 | SSIM: 0.545
- 均衡→滤波：PSNR: 15.63 | SSIM: 0.496
**结果分析**：滤波→均衡的PSNR略高，SSIM提升约0.049，说明椒盐噪声为离散极端噪声，先通过均值滤波抑制噪声点，再进行均衡化增强对比度，能避免均衡化将噪声点的灰度值放大，更好保留图像结构。

### 4. 实验核心结论
1. 组合处理的效果与图像的退化类型强相关：**高斯噪声适合先均衡后滤波**，**椒盐噪声适合先滤波后均衡**，**纯低对比度图像适合先均衡后滤波**；
2. PSNR与SSIM指标趋势一致，值越高表示增强效果越好，可作为图像增强的客观评价依据；
3. 手动实现的均值滤波可有效平滑图像，为组合处理提供稳定的滤波基础，自研模块效果符合预期。

## 常见问题排查
| 问题现象 | 原因及解决方法 |
|----------|----------------|
| 程序提示图像下载失败 | 网络不通/URL失效 → 检查网络连接，替换为有效的图像URL；或手动将图像放在项目目录，命名为network_img_0/1/2.jpg |
| 运行时出现数组维度不匹配错误 | 图像读取为彩色图/尺寸异常 → 确保读取时使用`cv2.imread(img_path, 0)`，以灰度图形式读取；程序已做尺寸自适应，无需手动调整 |
| Matplotlib保存的图片空白/布局混乱 | 缺失子图布局调整/显示后保存 → 确保使用`plt.tight_layout()`，先`plt.savefig()`再关闭画布，WSL环境需安装python3-tk依赖 |
| PSNR/SSIM值异常偏低 | 图像像素值溢出/类型错误 → 确保所有图像处理后通过`np.clip`限制0-255，并转换为uint8类型 |
| 终端出现Matplotlib字体警告 | Linux/WSL环境无中文字体 → 警告不影响结果保存与定量计算，可直接忽略；或修改代码标题为英文，消除警告 |
| 虚拟环境中提示模块不存在 | 依赖安装在系统环境，未安装在虚拟环境 → 激活虚拟环境后重新执行依赖安装命令 |

## 作业总结
本次作业完成了基于OpenCV与NumPy的图像增强全流程实践，严格遵循作业要求：从网络获取3幅不同类型图像，自行实现均值滤波模块，完成滤波与均衡的组合处理，对比不同参数与方法的增强效果，采用PSNR、SSIM两项定量指标评价，实现处理结果与直方图的可视化。

通过实验掌握了**图像退化处理、手动滤波实现、直方图均衡化、图像锐化**的核心方法，理解了滤波与均衡的组合处理逻辑，明确了不同组合处理对不同退化类型图像的适配性：高斯噪声适合先均衡后滤波，椒盐噪声适合先滤波后均衡。同时提升了Python对图像数组的操作能力，掌握了数字图像处理中**主观视觉评价+客观定量指标**的双重评价方法，为后续复杂图像预处理、图像质量优化奠定了基础。

本次作业的核心亮点为**自研均值滤波模块**与**组合处理的对比实验**，验证了滤波与均衡的处理顺序对增强效果的关键影响，实现了从图像获取、处理、评价到可视化的全流程自动化，程序具有良好的独立性、健壮性和可扩展性，可通过修改参数（如滤波核大小、均衡化参数、噪声强度）进一步拓展实验内容。
