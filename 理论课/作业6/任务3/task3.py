import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== 任务3：RANSAC 剔除错误匹配 =====================
# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取灰度图
img_box = cv2.imread('box.jpg', cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread('box_in_scene.jpg', cv2.IMREAD_GRAYSCALE)

# 检查图像是否读取成功
if img_box is None or img_scene is None:
    raise FileNotFoundError("请确保 box.jpg 和 box_in_scene.jpg 在当前目录！")

# 2. ORB 特征检测
orb = cv2.ORB_create(nfeatures=1000)
kp_box, des_box = orb.detectAndCompute(img_box, None)
kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

# 3. 暴力匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_box, des_scene)
matches = sorted(matches, key=lambda x: x.distance)

# 4. 提取匹配点坐标
src_pts = np.float32([kp_box[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 5. RANSAC 计算单应矩阵
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

# 6. 输出结果
print("===== 任务3 RANSAC 剔除误匹配结果 =====")
print(f"初始总匹配数：{len(matches)}")
print(f"RANSAC 正确匹配（内点）数量：{sum(matchesMask)}")
print("单应矩阵 H：\n", H)

# 7. 绘制 RANSAC 滤波后的匹配结果
img_ransac_result = cv2.drawMatches(
    img_box, kp_box, img_scene, kp_scene,
    matches, None, matchesMask=matchesMask, flags=2
)

# 8. 保存图片（已改名）
cv2.imwrite('orb_ransac_filtered_matches.png', img_ransac_result)

# 9. 显示图片
plt.figure(figsize=(16, 6))
plt.imshow(img_ransac_result)
plt.title('RANSAC 剔除误匹配结果')
plt.axis('off')
plt.show()