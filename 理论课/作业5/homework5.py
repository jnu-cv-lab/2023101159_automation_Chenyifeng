import cv2
import numpy as np
import os

# ===================== 路径配置（当前目录直接运行） =====================
test_first_path = "test_first.jpg"
test_final_path = "test_final.jpg"

# ===================== 1. 生成测试图：矩形、圆、平行线、垂直线 =====================
def make_test_img(size=600):
    img = np.ones((size, size, 3), np.uint8) * 255

    # 外框
    cv2.line(img, (50, 50),  (550, 50),  (0, 0, 0), 3)
    cv2.line(img, (50, 550), (550, 550), (0, 0, 0), 3)
    cv2.line(img, (50, 50),  (50, 550), (0, 0, 0), 3)
    cv2.line(img, (550, 50), (550, 550), (0, 0, 0), 3)

    # 圆
    cv2.circle(img, (220, 300), 130, (0, 0, 0), 3)

    # 正方形
    cv2.rectangle(img, (380, 170), (480, 430), (0, 0, 0), 3)

    # 平行线
    for y in [120, 220, 320, 420]:
        cv2.line(img, (80, y), (520, y), (80, 80, 80), 2)
    for x in [120, 220, 320, 420]:
        cv2.line(img, (x, 80), (x, 520), (80, 80, 80), 2)

    return img

# 生成原图
ori = make_test_img()
cv2.imwrite(test_first_path, ori)
print("✅ test_first.jpg 已保存")

h, w = ori.shape[:2]

# ===================== 2. 三种变换 =====================
# 相似变换
M_sim = cv2.getRotationMatrix2D((w/2, h/2), 15, 0.85)
img_sim = cv2.warpAffine(ori, M_sim, (w, h), borderValue=(255,255,255))

# 仿射变换
pts1 = np.float32([[50, 50], [550, 50], [50, 550]])
pts2 = np.float32([[90, 110], [510, 70], [70, 490]])
M_aff = cv2.getAffineTransform(pts1, pts2)
img_aff = cv2.warpAffine(ori, M_aff, (w, h), borderValue=(255,255,255))

# 透视变换
pts1_p = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
pts2_p = np.float32([[50, 40], [w-40, 30], [30, h-40], [w-30, h-50]])
M_per = cv2.getPerspectiveTransform(pts1_p, pts2_p)
img_per = cv2.warpPerspective(ori, M_per, (w, h), borderValue=(255,255,255))

# 拼接四宫格对比图
row1 = np.hstack((ori, img_sim))
row2 = np.hstack((img_aff, img_per))
img_all = np.vstack((row1, row2))

# 保存多张对比图（满足作业多图展示要求）
cv2.imwrite("transform_相似变换.jpg", img_sim)
cv2.imwrite("transform_仿射变换.jpg", img_aff)
cv2.imwrite("transform_透视变换.jpg", img_per)
cv2.imwrite("transform_四宫格对比.jpg", img_all)
print("所有变换对比图已保存")

# ===================== 3. A4 透视畸变校正 =====================
img = cv2.imread(test_final_path)
if img is None:
    print("未找到 test_final.jpg 图片")
    exit()

img_show = img.copy()
H, W = img.shape[:2]

# 预处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)
edge = cv2.Canny(blur, 30, 120)
edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

# 找纸张轮廓
cnts = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
paper_box = None

for c in cnts:
    arc = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * arc, True)
    if len(approx) == 4 and cv2.contourArea(approx) > W*H*0.2:
        paper_box = approx.reshape(4, 2).astype(np.float32)
        break

# 四点排序
def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

if paper_box is not None:
    paper_box = order_points(paper_box)
    cv2.polylines(img_show, [paper_box.astype(np.int32)], True, (0,0,255), 4)
else:
    paper_box = np.float32([[W*0.05,H*0.05],[W*0.95,H*0.05],[W*0.95,H*0.95],[W*0.05,H*0.95]])

cv2.imwrite("detect_纸张轮廓.jpg", img_show)

# 校正
A4_w, A4_h = 700, int(700 * 1.4142)
dst_pts = np.float32([[0,0],[A4_w,0],[A4_w,A4_h],[0,A4_h]])
M = cv2.getPerspectiveTransform(paper_box, dst_pts)
result = cv2.warpPerspective(img, M, (A4_w, A4_h))

cv2.imwrite("correct_final_校正完成图.jpg", result)
print("A4校正完成")