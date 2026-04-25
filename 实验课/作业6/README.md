# 基于 OpenCV 的局部特征检测、描述与图像匹配实验
- 学号：2023101159 姓名：陈亿锋
> ORB 特征匹配 + RANSAC 剔除误匹配 + 单应矩阵目标定位 + 参数对比 + SIFT 对比实验

## 一、实验环境
- Python + OpenCV
- 图片：`box.png`、`box_in_scene.png`

## 二、实验内容
### 1. 任务1：ORB 特征点检测
- 使用 `cv2.ORB_create(nfeatures=1000)` 检测特征点
- 计算关键点与 256 维二进制描述子
- 可视化特征点并输出关键点数量、描述子维度

### 2. 任务2：ORB 特征匹配
- 暴力匹配 `BFMatcher` + `NORM_HAMMING`
- 按匹配距离排序，显示前 50 个匹配
- 输出总匹配数

### 3. 任务3：RANSAC 剔除误匹配
- 用 `cv2.findHomography` + `RANSAC` 估计单应矩阵
- 按 mask 保留内点，剔除外点
- 输出内点数量、内点比例、单应矩阵 H

### 4. 任务4：目标定位
- 投影模板四角点到场景图
- 用 `polylines` 绘制定位框
- 验证目标是否成功定位

### 5. 任务6：nfeatures 参数对比
测试：500 / 1000 / 2000
对比指标：关键点数量、匹配数、内点比例、定位成功率

### 6. 选做：SIFT 与 ORB 对比
- SIFT：`SIFT_create` + KNN 匹配 + Lowe 比率测试
- ORB：二进制描述子 + 汉明距离
- 对比：匹配质量、内点比例、速度、稳定性

## 三、输出文件
- `box_keypoints.png`
- `box_in_scene_keypoints.png`
- `orb_match_result.png`
- `orb_ransac_filtered_matches.png`
- `target_localization_result.png`
- `sift_compare_*.png` 系列对比图

## 四、核心结论
1. **特征点**集中在角点、文字、纹理丰富区域
2. **ORB**：二进制描述子、快、用汉明距离
3. **SIFT**：浮点描述子、稳、用欧氏距离
4. **RANSAC**：按几何一致性剔除误匹配
5. **Homography**：适用于平面物体的透视变换
6. 特征点并非越多越好，过多会引入误匹配


