# C++ 视觉开发环境搭建与图像基本读写
- 作者：陈亿锋
- 学号：2023101159
- 专业：自动化

## 项目概述
本作业基于 **C++ 与 OpenCV** 实现数字图像处理基础实践，完成图像读取、基础信息打印、灰度化处理、指定像素值读取、图像区域裁剪、结果保存与窗口可视化显示等完整功能。通过本次实践掌握 Linux 环境下 C++ 编译、OpenCV 库链接、图像基础操作逻辑，理解数字图像在内存中的存储形式、通道结构与像素访问方式，为后续滤波、特征检测、目标跟踪等高级视觉任务打下基础。

## 技术栈
- 编程语言：C++11 及以上
- 核心库：OpenCV 4（C++ 版本）
- 编译工具：g++ / gcc
- 调试工具：gdb
- 运行环境：Ubuntu / WSL2（Linux 子系统）

## 文件清单
| 文件名 | 类型 | 功能说明 |
|--------|------|----------|
| main.cpp | 源代码文件 | 主程序：图像读取、信息输出、灰度化、像素读取、裁剪、保存、窗口显示 |
| test.jpg | 输入文件 | 测试原图，程序处理的输入源 |
| gray_output.jpg | 输出文件 | 程序自动生成的灰度图像 |
| cropped_output.jpg | 输出文件 | 程序自动生成的左上角 200×200 裁剪图像 |
| README.md | 文档文件 | 项目说明、环境配置、运行步骤、功能说明 |
| .vscode/ | 配置目录 | VS Code 调试与编译配置文件 |

## 环境配置
### 1. 环境要求
- 系统：Linux（Ubuntu 18.04+）/ Windows WSL2
- 编译器：g++（已预装在 Ubuntu/WSL）
- OpenCV：C++ 开发版

### 2. 依赖安装
执行以下命令安装 OpenCV C++ 开发库：
```bash
sudo apt update
sudo apt install libopencv-dev -y
```

验证安装：
```bash
pkg-config --modversion opencv4
```
出现版本号即安装成功。

## 编译与运行步骤
### 1. 进入项目目录
```bash
cd /home/gevle/cv-course
```

### 2. 创建 build 目录
```bash
mkdir -p build
```

### 3. 编译代码
```bash
g++ main.cpp -o build/opencv_demo `pkg-config --cflags --libs opencv4`
```

### 4. 运行程序
```bash
./build/opencv_demo test.jpg
```

### 5. 运行成功输出示例
```
=== 图像基本信息 ===
宽度: 600 像素
高度: 400 像素
通道数: 3
像素数据类型: 8位无符号，3通道（彩色）
✅ 灰度图已保存为: gray_output.jpg
像素(100,100)的 BGR 值: 37, 58, 66
✅ 左上角 200x200 区域已保存为: cropped_output.jpg
```

## 核心功能解析
### 1. 图像读取与合法性判断
```cpp
Mat img = imread(argv[1], IMREAD_COLOR);
if (img.empty()) {
    cout << "无法读取图片，请检查路径！" << endl;
    return -1;
}
```
作用：判断图像是否成功加载，避免程序崩溃。

### 2. 输出图像基础信息
```cpp
cout << "宽度: " << img.cols << endl;
cout << "高度: " << img.rows << endl;
cout << "通道数: " << img.channels() << endl;
```
输出图像尺寸、通道数、数据类型，是图像处理最基础操作。

### 3. 彩色图像转灰度图
```cpp
Mat gray_img;
cvtColor(img, gray_img, COLOR_BGR2GRAY);
```
将 3 通道彩色图转为 1 通道灰度图，用于后续简化处理。

### 4. 读取指定位置像素值
```cpp
Vec3b pixel = img.at<Vec3b>(100, 100);
cout << "像素(100,100)的 BGR 值: " 
     << (int)pixel[0] << ", " 
     << (int)pixel[1] << ", " 
     << (int)pixel[2] << endl;
```
OpenCV 默认为 **BGR** 通道顺序，可直接访问像素值。

### 5. 图像区域裁剪
```cpp
Rect roi(0, 0, 200, 200);
Mat cropped_img = img(roi);
imwrite("cropped_output.jpg", cropped_img);
```
裁剪左上角 200×200 区域并保存。

### 6. 图像显示与保存
```cpp
imshow("原图", img);
imshow("灰度图", gray_img);
waitKey(0);
destroyAllWindows();
```
弹出图像窗口，按键后关闭。

## VS Code 调试说明
本项目已配置：
- `launch.json`：调试程序路径
- `tasks.json`：自动编译并链接 OpenCV

可直接按 **F5** 启动调试。

## 常见问题排查
| 问题 | 解决方法 |
|------|----------|
| 找不到 opencv2/opencv.hpp | 未安装 libopencv-dev，执行安装命令 |
| 编译报错 | 必须加上 `pkg-config --cflags --libs opencv4` |
| 提示用法: ./opencv_demo <图片路径> | 运行时未加 test.jpg 参数 |
| 窗口无法显示 | 在 WSL 内运行需安装图形界面支持 |

## 作业总结
本次作业成功完成：
✅ C++ OpenCV 环境搭建
✅ 图像读取、信息输出
✅ 灰度化、像素访问、图像裁剪
✅ 结果保存与窗口可视化
✅ VS Code 编译 + 调试配置

掌握了 Linux 下 C++ 图像处理的完整流程，为后续视觉算法开发奠定基础。

告诉我就行！
