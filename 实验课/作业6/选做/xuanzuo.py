import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# ===================== 选做：SIFT 与 ORB 特征匹配对比实验 =====================
# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取灰度图像
img_box = cv2.imread('box.jpg', cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread('box_in_scene.jpg', cv2.IMREAD_GRAYSCALE)

# 图像读取校验
if img_box is None or img_scene is None:
    raise FileNotFoundError("错误：请确保图片在当前目录！")

# ---------------------- SIFT 特征检测与匹配 ----------------------
print("===== SIFT 特征匹配 =====")
start_time = time.time()

# 初始化 SIFT
sift = cv2.SIFT_create()
kp_sift_box, des_sift_box = sift.detectAndCompute(img_box, None)
kp_sift_scene, des_sift_scene = sift.detectAndCompute(img_scene, None)

print(f"模板关键点: {len(kp_sift_box)}")
print(f"场景关键点: {len(kp_sift_scene)}")

# KNN 匹配 + 比率测试
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches_knn = bf.knnMatch(des_sift_box, des_sift_scene, k=2)

good_matches = []
for m, n in matches_knn:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"优质匹配数: {len(good_matches)}")

# RANSAC 计算单应矩阵
located_sift = False
if len(good_matches) >= 4:
    src_pts = np.float32([kp_sift_box[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_sift_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H_sift, mask_sift = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    inlier_count_sift = int(np.sum(mask_sift))
    inlier_ratio_sift = inlier_count_sift / len(good_matches)
    
    # 目标定位
    h, w = img_box.shape
    box_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    scene_corners_sift = cv2.perspectiveTransform(box_corners, H_sift)
    located_sift = True
else:
    inlier_count_sift = 0
    inlier_ratio_sift = 0.0
    scene_corners_sift = None

time_sift = time.time() - start_time
print(f"RANSAC内点: {inlier_count_sift} | 内点比例: {inlier_ratio_sift:.4f}")
print(f"定位成功: {'是 ✅' if located_sift else '否 ❌'} | 耗时: {time_sift:.4f}s")

# 保存 SIFT 匹配图
img_sift_matches = cv2.drawMatches(img_box, kp_sift_box, img_scene, kp_sift_scene, good_matches[:50], None, flags=2)
cv2.imwrite('sift_compare_matches.png', img_sift_matches)

# ---------------------- ORB 特征检测与匹配 ----------------------
print("\n===== ORB 特征匹配 =====")
start_time = time.time()

orb = cv2.ORB_create(nfeatures=1000)
kp_orb_box, des_orb_box = orb.detectAndCompute(img_box, None)
kp_orb_scene, des_orb_scene = orb.detectAndCompute(img_scene, None)

print(f"模板关键点: {len(kp_orb_box)}")
print(f"场景关键点: {len(kp_orb_scene)}")

# 暴力匹配
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf_orb.match(des_orb_box, des_orb_scene)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)
match_count_orb = len(matches_orb)

print(f"总匹配数: {match_count_orb}")

# RANSAC
src_pts_orb = np.float32([kp_orb_box[m.queryIdx].pt for m in matches_orb]).reshape(-1, 1, 2)
dst_pts_orb = np.float32([kp_orb_scene[m.trainIdx].pt for m in matches_orb]).reshape(-1, 1, 2)
H_orb, mask_orb = cv2.findHomography(src_pts_orb, dst_pts_orb, cv2.RANSAC, 5.0)

inlier_count_orb = int(np.sum(mask_orb))
inlier_ratio_orb = inlier_count_orb / match_count_orb
scene_corners_orb = cv2.perspectiveTransform(box_corners, H_orb)
located_orb = True

time_orb = time.time() - start_time
print(f"RANSAC内点: {inlier_count_orb} | 内点比例: {inlier_ratio_orb:.4f}")
print(f"定位成功: {'是 ✅' if located_orb else '否 ❌'} | 耗时: {time_orb:.4f}s")

# 保存 ORB 匹配图
img_orb_matches = cv2.drawMatches(img_box, kp_orb_box, img_scene, kp_orb_scene, matches_orb[:50], None, flags=2)
cv2.imwrite('orb_compare_matches.png', img_orb_matches)

# ---------------------- SIFT 定位结果可视化 ----------------------
img_scene_color = cv2.cvtColor(img_scene, cv2.COLOR_GRAY2BGR)
img_sift_local = cv2.polylines(img_scene_color.copy(), [np.int32(scene_corners_sift)], 
                               True, (0, 255, 0), 3)
cv2.imwrite('sift_compare_localization.png', img_sift_local)

# ---------------------- ORB 定位结果可视化 ----------------------
img_orb_local = cv2.polylines(img_scene_color.copy(), [np.int32(scene_corners_orb)], 
                             True, (0, 0, 255), 3)
cv2.imwrite('orb_compare_localization.png', img_orb_local)

# ---------------------- 最终对比表格 ----------------------
print("\n" + "=" * 70)
print("              SIFT vs ORB 对比结果")
print("=" * 70)
print(f"{'方法':<6}{'匹配数':<8}{'内点数':<10}{'内点比例':<12}{'定位':<10}{'耗时(s)':<10}{'速度'}")
print("-" * 70)
print(f"SIFT   {len(good_matches):<8}{inlier_count_sift:<10}{inlier_ratio_sift:.4f}{'✅' if located_sift else '❌':<10}{time_sift:<10.4f}慢")
print(f"ORB    {match_count_orb:<8}{inlier_count_orb:<10}{inlier_ratio_orb:.4f}{'✅' if located_orb else '❌':<10}{time_orb:<10.4f}快")
print("=" * 70)
