import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== 任务2：ORB 初始暴力匹配 =====================
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取灰度图像
img_box = cv2.imread('box.jpg', cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread('box_in_scene.jpg', cv2.IMREAD_GRAYSCALE)

# 判断图像是否读取成功
if img_box is None or img_scene is None:
    raise FileNotFoundError("错误：请确保图片在代码同一目录下！")

# 2. 创建 ORB 特征检测器
orb = cv2.ORB_create(nfeatures=1000)

# 3. 检测关键点并计算描述子
kp_box, des_box = orb.detectAndCompute(img_box, None)
kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

# 4. 创建暴力匹配器（ORB 必须用汉明距离 NORM_HAMMING）
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 5. 匹配描述子
matches = bf.match(des_box, des_scene)

# 6. 按匹配距离从小到大排序（距离越小，匹配越准确）
matches = sorted(matches, key=lambda x: x.distance)

# 7. 输出结果
print("===== 任务2 ORB 初始匹配 =====")
print(f"总匹配数量：{len(matches)}")

# 8. 绘制前 50 个最佳匹配
img_matches = cv2.drawMatches(img_box, kp_box, img_scene, kp_scene, matches[:50], None, flags=2)

# 9. 保存结果图片
cv2.imwrite('orb_match_result.png', img_matches)

# 10. 显示匹配图
plt.figure(figsize=(16, 6))
plt.imshow(img_matches)
plt.title('ORB 初始暴力匹配结果（前50个）')
plt.axis('off')
plt.show()