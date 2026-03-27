# Python视觉开发环境搭建与图像基本读写

- 作者：陈亿锋
- 学号：2023101159
- 专业：自动化

## 项目概述
本作业围绕OpenCV库展开数字图像处理基础实践，核心实现了图像的读取、基础信息解析、BGR-RGB色彩空间转换、灰度化处理、指定坐标像素值读取、图像区域裁剪、结果保存与可视化图像展示等功能。通过本次作业，掌握Python环境下OpenCV库的基本使用方法，了解数字图像的存储格式（BGR/RGB）、像素组成及基础处理逻辑，为复杂图像处理（如几何变换、滤波去噪、边缘检测、特征提取）奠定基础。

## 技术栈
编程语言：Python 3.8+（兼容3.7及以上版本）
核心库：
- OpenCV-Python（cv2）：图像读取、处理、保存
- NumPy：图像存储操作、像素值处理
- Matplotlib：图像可视化展示

## 文件清单
| 文件名 | 类型 | 功能说明 |
|--------|------|----------|
| main.py | 源代码文件 | 主程序入口，包含所有图像处理逻辑：图片读取、信息打印、色彩转换、灰度化、像素值读取、裁剪、显示、保存 |
| test.jpg | 输入文件 | 原始图像测试，作为程序处理的输入源（建议使用jpg/png格式，避免特殊编码格式） |
| gray_test.jpg | 输出文件 | 程序运行后自动生成灰度化处理结果图，保留原始图像尺寸，仅保留灰度信息 |
| cropped_test.jpg | 输出文件 | 程序运行后自动生成的裁剪结果图，截取原始图像左上角100×100像素区域（图像尺寸足够时生成） |
| README.md | 文档文件 | 项目完整说明，包含环境配置、运行步骤、功能解析、常见问题等 |

## 环境配置
### 1、环境要求
操作系统：Windows 10/11、Linux（Ubuntu 18.04+）、macOS 12+
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
将img_process.py和test.jpg放在同一目录下，避免路径读取错误（若文件在src/目录下，需同步修改代码中图像读取路径为"src/test.jpg"）；
确认test.jpg文件未损坏，可通过系统图片查看器正常打开。

### 2、运行程序
#### 方式一：终端/命令行运行（推荐）
打开终端/命令提示符，切换到文件所在目录（示例：Windows系统）：
```bash
cd D:\homework\2023101159-automation-chenyifeng-homework
```
执行运行命令：
```bash
# Windows/Linux/macOS 通用
python img_process.py
# 若系统存在多个 Python 版本，指定 Python3
python3 img_process.py
```

#### 方式二：IDE运行（VS Code/PyCharm）
打开IDE并导入项目目录；
右键点击img_process.py文件，选择「运行」/「Run Python File」；
等待程序执行，自动弹出图像显示窗口。

### 3. 运行结果
程序执行后会完成以下操作：
- 控制台输出：打印图像宽度、高度、通道数、像素数据类型，灰度图指定坐标像素值（图像尺寸足够时），文件保存提示；
- 图像可视化：弹出包含「原始图像、灰度图」的显示窗口（无坐标轴干扰）；
- 文件生成：在项目目录下自动生成gray_test.jpg（灰度图），图像尺寸足够时额外生成cropped_test.jpg（100×100裁剪图）。

## 核心功能解析
### 1. 图像读取与有效性验证
```python
import cv2
# 读取图像（OpenCV 默认 BGR 格式）
img = cv2.imread("test.jpg")
# 校验读取结果，避免文件不存在/损坏导致程序崩溃
if img is None:
    print("无法读取图片，请检查文件名和路径！")
    exit()
```
核心函数：cv2.imread()返回图像的NumPy数组存储，None表示读取失败；
关键作用：提升程序健壮性，避免因文件路径/格式问题导致后续代码崩溃。

### 2. 基础图像信息解析
```python
# 获取图像尺寸（高, 宽, 通道数）
height, width, channels = img.shape
# 获取像素数据类型（通常为 uint8）
dtype = img.dtype
print("图像基本信息：")
print(f"宽度: {width} 像素")
print(f"高度: {height} 像素")
print(f"通道数: {channels}")
print(f"像素数据类型: {img.dtype}")
```
坐标规则：OpenCV中图像shape返回顺序为(高度, 宽度, 通道数)，像素坐标为(行, 列)，对应图像的(y, x)轴，而非常规(x, y)。

### 3. 色彩空间转换与图像可视化
```python
import matplotlib.pyplot as plt
# BGR 转 RGB（适配 Matplotlib 显示规则）
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 显示原图（隐藏坐标轴）
plt.figure("原图")
plt.imshow(img_rgb)
plt.axis("off")
```
核心问题：OpenCV读取图像默认为BGR格式，而Matplotlib默认按RGB格式显示，直接显示会导致色彩失真；
解决方法：通过cv2.cvtColor(img, cv2.COLOR_BGR2RGB)完成通道顺序转换，保证图像正常显示。

### 4. 灰度化处理与像素值读取
```python
# 彩色图转灰度图（通道数从 3 降为 1）
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 显示灰度图（隐藏坐标轴）
plt.figure("灰度图")
plt.imshow(gray_img, cmap="gray")
plt.axis("off")
# 保存灰度图
cv2.imwrite("gray_test.jpg", gray_img)
print("灰度图已保存为: gray_test.jpg")

# 读取灰度图指定坐标像素值（判断图像尺寸是否足够）
if gray_img.shape[0] > 100 and gray_img.shape[1] > 100:
    pixel_val = gray_img[100, 100]
    print(f"灰度图 (100,100) 位置像素值: {pixel_val}")
else:
    print("图像太小，无法取 (100,100) 位置像素")
```
灰度化原理：通过加权平均RGB通道值（Y=0.299R+0.587G+0.114B）将3通道彩色图转为1通道灰度图，仅保留亮度信息；
像素读取规则：gray_img[y, x]可读取指定坐标像素值，需先判断图像尺寸避免索引越界。

### 5. 图像裁剪与保存
```python
crop_size = 100
# 判断图像尺寸是否满足裁剪条件
if height >= crop_size and width >= crop_size:
    # 裁剪左上角 100×100 区域（切片规则：行起始:行结束, 列起始:列结束）
    cropped = img[0:crop_size, 0:crop_size]
    cv2.imwrite("cropped_test.jpg", cropped)
    print(f"左上角 {crop_size}x{crop_size} 区域已保存为: cropped_test.jpg")
else:
    print(f"图像小于 {crop_size}x{crop_size}，无法裁剪")
```
裁剪规则：NumPy数组切片img[y1:y2, x1:x2]实现图像区域裁剪，左闭右开区间，0:100 对应100×100像素区域；
容错处理：先判断图像尺寸是否满足裁剪要求，避免切片索引越界报错。

### 6. 图像显示窗口触发
```python
# 显示所有图像窗口
plt.show()
```
关键作用：plt.show()会阻塞程序直至关闭所有图像窗口，保证可视化效果正常展示。

## 常见问题排查
| 问题现象 | 原因及解决方法 |
|----------|----------------|
| 程序提示「无法读取图片」 | 文件路径错误/名称大小写问题/格式不支持 → 确认文件同目录、名称一致、使用 jpg/png 格式 |
| 图像显示色彩失真 | 未转换色彩空间 → 显示前执行cv2.cvtColor(img, cv2.COLOR_BGR2RGB) |
| 提示「图像太小，无法取 (100,100) 位置像素」 | 测试图像尺寸不足100×100 → 更换更大尺寸的test.jpg |
| 提示「图像小于 100x100，无法裁剪」 | 测试图像尺寸不足100×100 → 更换更大尺寸的test.jpg |
| 安装库时报错 | 网络问题 → 使用国内镜像源；Python版本过低 → 升级到3.7+ |
| 图像窗口无响应 | 系统图形界面异常 → 关闭所有窗口后重新启动程序 |

## 作业总结
本次作业完成了OpenCV图像处理基础实践，掌握了图像读取、基础信息解析、色彩空间转换、灰度化、像素值读取、图像裁剪、结果保存与可视化等核心操作，了解了数字图像的存储格式与可视化规则，以及代码容错处理（如尺寸判断、读取验证）的重要性。
