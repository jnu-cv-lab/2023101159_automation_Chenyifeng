# 羽毛球击球动作识别实验 - README.md
```markdown
# 作业11：基于MediaPipe Pose与Transformer的羽毛球击球动作识别

## 📂 项目结构
```
作业11/
├── dataset_npy/                  # 预处理后的数据与模型权重
│   ├── X_train.npy               # 训练集骨架特征
│   ├── X_test.npy                # 测试集骨架特征
│   ├── y_train.npy               # 训练集标签
│   ├── y_test.npy                # 测试集标签
│   ├── label_map.json            # 动作类别映射
│   └── badminton_transformer.pth# 训练好的模型权重
├── output_images/                # 实验结果可视化文件
│   ├── training_curves.png       # 训练损失与准确率曲线
│   ├── confusion_matrix.png      # 模型混淆矩阵
│   ├── attention_heatmap.png     # Transformer注意力热力图
│   ├── skeleton_video_1.mp4      # 动作1骨架可视化视频
│   └── skeleton_video_2.mp4      # 动作2骨架可视化视频
├── zy11.py                       # 完整实验代码
├── 实验报告11.pdf                # 实验报告文档
└── README.md                     # 项目说明文件
```

---

## 🎯 实验目标
本项目以羽毛球6类击球动作识别为任务，实现一套**视频骨架提取 + 时序Transformer分类**的完整流程，覆盖实验全部要求：
1.  视频读取、MediaPipe Pose人体关键点提取、时序重采样、骨架归一化
2.  数据集构建、Transformer模型训练、测试评估与单样本推理
3.  骨架可视化、注意力热力图分析等拓展功能

---

## ⚙️ 运行环境与依赖
- 运行平台：WSL / Linux / Windows（Python 3.12）
- 核心依赖库：
  ```bash
  pip install opencv-python mediapipe torch numpy scikit-learn matplotlib
  ```

---

## 🚀 运行说明
1.  **数据准备**：
    - 在代码同级目录创建 `./archive` 文件夹，内部按类别存放羽毛球视频文件
    - 文件夹结构示例：
      ```
      archive/
      ├── forehand_drive/
      ├── forehand_lift/
      ├── forehand_net_shot/
      ├── forehand_clear/
      ├── backhand_drive/
      └── backhand_net_shot/
      ```

2.  **首次运行**：
    ```bash
    python zy11.py
    ```
    - 自动解析视频、提取骨架、生成`.npy`数据文件
    - 训练模型并保存权重、训练曲线、混淆矩阵

3.  **后续运行**：
    - 代码会直接加载`dataset_npy/`下的预处理数据，跳过视频解析，大幅提升运行速度
    - 如需重新处理视频，删除`dataset_npy/`下的`.npy`文件即可

4.  **骨架可视化与推理**：
    - 修改主函数中`vid1`和`vid2`的路径为你数据集中的视频文件
    - 运行后会自动生成骨架可视化视频与注意力热力图

---

## 📊 实验结果
- **模型性能**：测试集整体准确率 **51.02%**，反手网前球（backhand_net_shot）识别效果最优（F1=0.6909）
- **关键输出文件**：
  - `training_curves.png`：展示训练损失与测试准确率变化趋势
  - `confusion_matrix.png`：直观呈现各类动作的混淆情况
  - `skeleton_video_*.mp4`：两段动作的骨架可视化视频，验证姿态提取效果
  - `attention_heatmap.png`：Transformer注意力热力图，展示模型关注的关键帧

---

## 📌 核心模块说明
1.  **数据预处理**：
    - 提取人体33个关键点（132维特征）
    - 线性插值重采样为30帧，统一时序长度
    - 髋中心中心化+肩宽归一化，消除拍摄距离与体型干扰

2.  **模型结构**：
    - 骨架特征嵌入 + 正弦位置编码
    - 2层自定义Transformer编码器（支持注意力权重提取）
    - 均值池化 + 分类头，输出6类动作预测结果

3.  **拓展功能**：
    - 视频骨架绘制与可视化
    - 多头注意力热力图分析，增强模型可解释性

---

## 📝 问题与优化方向
- **现存问题**：样本量不足、动作相似度高导致整体准确率偏低，部分类别混淆严重，训练后期存在轻微过拟合
- **优化方案**：扩充数据集、增加数据增强、添加关节运动速度/角度等衍生特征、调整模型正则化策略

---

## 📜 提交项核验
✅ 预处理代码：视频读取、关键点提取、重采样、归一化、npy保存
✅ 训练代码：Dataset、DataLoader、模型、训练循环完整实现
✅ 测试与推理：准确率、混淆矩阵、单样本推理结果输出
✅ 实验报告：方法说明、模型结构、训练曲线、结果分析齐全
✅ 拓展项：双视频骨架可视化 + 注意力热力图分析
```
