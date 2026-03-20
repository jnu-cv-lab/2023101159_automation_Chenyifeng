import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("test.jpg")
if img is None:
    print("无法读取图片，请检查文件名和路径！")
    exit()
height, width, channels = img.shape
print("图像基本信息：")
print(f"宽度: {width} 像素")
print(f"高度: {height} 像素")
print(f"通道数: {channels}")
print(f"像素数据类型: {img.dtype}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure("原图")
plt.imshow(img_rgb)
plt.axis("off")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure("灰度图")
plt.imshow(gray_img, cmap="gray")
plt.axis("off")
cv2.imwrite("gray_test.jpg", gray_img)
print("灰度图已保存为: gray_test.jpg")
if gray_img.shape[0] > 100 and gray_img.shape[1] > 100:
    pixel_val = gray_img[100, 100]
    print(f"灰度图 (100,100) 位置像素值: {pixel_val}")
else:
    print("图像太小，无法取 (100,100) 位置像素")
crop_size = 100
if height >= crop_size and width >= crop_size:
    cropped = img[0:crop_size, 0:crop_size]
    cv2.imwrite("cropped_test.jpg", cropped)
    print(f"左上角 {crop_size}x{crop_size} 区域已保存为: cropped_test.jpg")
else:
    print(f"图像小于 {crop_size}x{crop_size}，无法裁剪")
plt.show()
