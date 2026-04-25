import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== 任务6：ORB 参数 nfeatures 对比实验 =====================
# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取灰度图像
img_box = cv2.imread('box.jpg', cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread('box_in_scene.jpg', cv2.IMREAD_GRAYSCALE)

# 图像读取校验
if img_box is None or img_scene is None:
    raise FileNotFoundError("错误：请确保 box.jpg 和 box_in_scene.jpg 在当前目录！")

# 2. 定义对比参数列表
nfeatures_list = [500, 1000, 2000]
results = []  # 存储实验数据

# 3. 遍历参数进行实验
for n in nfeatures_list:
    print(f"\n========== 实验参数：nfeatures = {n} ==========")
    
    # ORB 特征检测
    orb = cv2.ORB_create(nfeatures=n)
    kp_box, des_box = orb.detectAndCompute(img_box, None)
    kp_scene, des_scene = orb.detectAndCompute(img_scene, None)
    
    # 统计关键点数量
    num_kp_box = len(kp_box)
    num_kp_scene = len(kp_scene)
    print(f"模板图像关键点：{num_kp_box}")
    print(f"场景图像关键点：{num_kp_scene}")
    
    # 暴力匹配 + 排序
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_box, des_scene)
    matches = sorted(matches, key=lambda x: x.distance)
    num_matches = len(matches)
    print(f"初始匹配总数：{num_matches}")
    
    # RANSAC 计算单应矩阵
    src_pts = np.float32([kp_box[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # 统计内点
    num_inliers = int(np.sum(mask))
    inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0
    print(f"RANSAC 内点数量：{num_inliers}")
    print(f"内点比例：{inlier_ratio:.4f}")
    
    # 目标定位验证
    h, w = img_box.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    scene_corners = cv2.perspectiveTransform(corners, H)
    
    # 判断是否定位成功
    locate_success = True
    for (x, y) in scene_corners[:, 0, :]:
        if x < 0 or x >= img_scene.shape[1] or y < 0 or y >= img_scene.shape[0]:
            locate_success = False
            break
    print(f"目标定位结果：{'成功 ✅' if locate_success else '失败 ❌'}")
    
    # 保存匹配效果图
    img_match = cv2.drawMatches(img_box, kp_box, img_scene, kp_scene, matches[:50], None, flags=2)
    cv2.imwrite(f'orb_match_n{n}.png', img_match)
    print(f"匹配图已保存：orb_match_n{n}.png")
    
    # 记录结果
    results.append([
        n, num_kp_box, num_kp_scene, num_matches,
        num_inliers, round(inlier_ratio, 4), "成功" if locate_success else "失败"
    ])

# 4. 打印最终对比表格
print("\n" + "=" * 90)
print("                ORB 参数 nfeatures 对比实验结果表")
print("=" * 90)
header = f"{'nfeatures':<10}{'模板关键点':<12}{'场景关键点':<12}{'总匹配数':<10}{'内点数量':<10}{'内点比例':<10}{'定位结果'}"
print(header)
print("-" * 90)
for row in results:
    print(f"{row[0]:<10}{row[1]:<12}{row[2]:<12}{row[3]:<10}{row[4]:<10}{row[5]:<10}{row[6]}")
print("=" * 90)