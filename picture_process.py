import cv2
import numpy as np
import time

start = time.time()
img = cv2.imread("C:/Users/CXY/Desktop/3.jpg")
print("Read image")
# 设置新的分辨率
new_resolution = (200, 200)

# 调整图像大小并保存
img = cv2.resize(img, new_resolution)
r = img.shape
# roi区域
roi = img[0:int(r[0]), 0 :int(r[1])]

# 原图mask
mask = np.zeros(img.shape[:2], dtype=np.uint8)

# 矩形roi
rect = (0, 0, int(r[0]) - 1, int(r[1]) - 1)  # 包括前景的矩形，格式为(x,y,w,h)

bgdmodel = np.zeros((1, 65), np.float64)  # bg模型的临时数组
fgdmodel = np.zeros((1, 65), np.float64)  # fg模型的临时数组

cv2.grabCut(img, mask, rect, bgdmodel, fgdmodel, 5 , mode=cv2.GC_INIT_WITH_RECT)

# 提取前景和可能的前景区域
mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
print(mask2.shape)

result = cv2.bitwise_and(img, img, mask=mask2)
# 读取图像

# 设置新的分辨率
new_resolution = (100, 100)

# 调整图像大小并保存
result = cv2.resize(result, new_resolution)
roi = cv2.resize(roi, new_resolution)
end = time.time()
cv2.imwrite('result.jpg', result)
print("Time:{:.2f}".format(end-start))


def delete_background(path):
    img = cv2.imread(path)
    print("Read image")
    # 设置新的分辨率
    new_resolution = (200, 200)

    # 调整图像大小并保存
    img = cv2.resize(img, new_resolution)
    r = img.shape
    # roi区域
    roi = img[0:int(r[0]) - 1, 0:int(r[1]) - 1]

    # 原图mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # 矩形roi
    rect = (0, 0, int(r[0]) - 1, int(r[1]) - 1)  # 包括前景的矩形，格式为(x,y,w,h)

    bgdmodel = np.zeros((1, 65), np.float64)  # bg模型的临时数组
    fgdmodel = np.zeros((1, 65), np.float64)  # fg模型的临时数组

    cv2.grabCut(img, mask, rect, bgdmodel, fgdmodel, 5, mode=cv2.GC_INIT_WITH_RECT)

    # 提取前景和可能的前景区域
    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    print(mask2.shape)

    result = cv2.bitwise_and(img, img, mask=mask2)
    # 读取图像

    # 设置新的分辨率
    new_resolution = (100, 100)

    # 调整图像大小并保存
    result = cv2.resize(result, new_resolution)
    cv2.imwrite('result.jpg', result)