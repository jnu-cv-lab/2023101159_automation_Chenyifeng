# 实验作业4：图像下采样与抗混叠实验
**姓名**：陈亿锋
**学号**：2023101159

## 一、项目目的
1. 深入理解图像下采样过程中高频信号折叠引发**混叠**的物理本质，明确**抗混叠滤波**的必要性；
2. 验证高斯滤波经验公式 $\sigma \approx 0.45M$ 的正确性，掌握高斯滤波下采样的工程实现方法；
3. 通过傅里叶变换直观分析图像频域特征，验证滤波与下采样操作对图像频谱的影响；
4. 实现**自适应下采样算法**，实现图像细节区与平坦区的差异化下采样；
5. 通过误差热力图、MSE/PSNR/SSIM 量化指标，直观对比不同下采样方案的优劣。

## 二、运行环境
- 操作系统：Linux/Ubuntu
- 编程语言：Python3
- 核心依赖库：
  - OpenCV（图像处理）
  - NumPy（数值计算）
  - Matplotlib（可视化绘图）

### 依赖安装命令
```bash
pip install opencv-python numpy matplotlib
```

## 三、主要功能
1. **测试图像生成**：自动生成棋盘格图像、Chirp调频测试图，用于清晰观察下采样混叠现象；
2. **两种下采样方案实现**：
   - 直接下采样（无滤波，易产生混叠）
   - 高斯滤波+下采样（抗混叠，抑制高频混叠）
3. **频域分析**：对原图、直接下采样图、高斯下采样图计算并展示傅里叶频谱，对比混叠与抗混叠差异；
4. **自适应下采样**：基于图像梯度幅值，自动分配局部下采样倍数$M$和高斯滤波$\sigma$，细节区低倍数下采样、平坦区高倍数下采样；
5. **效果评价**：绘制误差热力图，计算MSE、PSNR、SSIM量化指标，对比自适应下采样与全局统一下采样效果。

## 四、核心代码说明
### 4.1 基础下采样与高斯滤波
```python
# 直接下采样：每隔M个像素取一个值
def downsample(img, M):
    return img[::M, ::M]

# 高斯滤波+下采样（抗混叠核心）
def gaussian_downsample(img, M, sigma):
    blurred = cv2.GaussianBlur(img, (5, 5), sigma)  # 高斯滤波抑制高频
    return downsample(blurred, M)
```

### 4.2 FFT频谱分析
```python
def get_fft_spectrum(img):
    f = np.fft.fft2(img)          # 二维傅里叶变换
    f_shift = np.fft.fftshift(f) # 低频分量移至频谱中心
    magnitude = 20 * np.log(np.abs(f_shift) + 1)  # 计算频谱幅值
    return magnitude
```

### 4.3 自适应参数生成（梯度计算）
```python
# 用Sobel算子计算图像梯度幅值
def compute_gradient(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    return grad_mag

# 由梯度生成局部下采样倍数M和高斯核σ
def generate_local_M_sigma(grad_mag, M_min=2, M_max=4):
    local_M = M_max - (M_max - M_min) * grad_mag
    local_sigma = 0.45 * local_M  # 遵循经验公式
    return local_M, local_sigma
```

### 4.4 自适应高斯滤波与下采样
```python
# 分块自适应高斯滤波
def adaptive_gaussian_blur(img, local_sigma):
    h, w = img.shape
    blurred = np.zeros_like(img, dtype=np.float32)
    block = 4
    for i in range(0, h, block):
        for j in range(0, w, block):
            sg = local_sigma[i:i+block, j:j+block].mean()
            ksize = (2*int(4*sg)+1, 2*int(4*sg)+1)
            blurred[i:i+block, j:j+block] = cv2.GaussianBlur(
                img[i:i+block, j:j+block], ksize, sg
            )
    return blurred.astype(np.uint8)

# 自适应下采样
def adaptive_downsample(img, local_M):
    h, w = img.shape
    new_h, new_w = h // 4, w // 4
    out = np.zeros((new_h, new_w), dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            y = i*4
            x = j*4
            M = int(round(local_M[y:y+4, x:x+4].mean()))
            out[i,j] = img[y:y+4, x:x+4][::M, ::M].mean()
    return out
```

### 4.5 图像质量评价指标
```python
def compute_metrics(original, upsampled):
    mse = np.mean((original - upsampled)**2)
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    ssim = compute_ssim(original, upsampled)
    return mse, psnr, ssim
```

## 五、核心参数说明
1. **全局下采样倍数**：$M=4$，图像宽高均缩小为原图的1/4，面积缩小为1/16；
2. **高斯核标准差**：$\sigma \approx 0.45 \times M$（实验验证核心经验公式）；
3. **梯度算子**：Sobel算子，计算图像梯度幅值，区分**细节区**（梯度大）和**平坦区**（梯度小）；
4. **参数映射**：梯度幅值归一化至[0,1]，线性映射得到局部下采样倍数：
   $$M = M_{max} - (M_{max}-M_{min}) \times grad$$
5. **评价指标**：
   - MSE（均方误差）：数值越小，图像误差越小；
   - PSNR（峰值信噪比）：数值越大，图像质量越高；
   - SSIM（结构相似性）：数值越接近1，图像结构越相似。

## 六、运行步骤
1. 进入项目目录：`cd cv-course/build/`
2. 创建代码文件：
   ```bash
   touch homework4.py
   touch homework4_1.py
   ```
3. 使用VScode编辑上述Python文件，写入实验代码；
4. 激活虚拟开发环境：
   ```bash
   source /home/lzy/cv-course/.venv-basic/bin/activate
   ```
5. 运行基础下采样实验脚本：
   ```bash
   python3 homework4.py
   ```
6. 运行自适应下采样实验脚本：
   ```bash
   python3 homework4_1.py
   ```

## 七、实验预期结果
1. 直接下采样会出现明显**混叠失真**，高频区域出现伪影；
2. 高斯滤波后下采样可有效**抑制混叠**，图像边缘更平滑；
3. 傅里叶频谱可直观看到：直接下采样高频分量混叠，滤波后高频被有效抑制；
4. 自适应下采样相比全局下采样，**MSE更低、PSNR/SSIM更高**，误差热力图显示失真更小，图像质量更优。
