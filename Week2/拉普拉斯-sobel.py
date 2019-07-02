# -*- coding: utf-8 -*-

# @Author: xyq

import cv2
import numpy as np

img = cv2.imread('th.jpg')
'''
二阶导 拉普拉斯核 双边缘效果
'''
# 二阶导 拉普拉斯核 ==》 边缘化,双边信息，图象好像变模糊了
kernel_lap = np.array([[0,1,0],[1,-4,1],[0,1,0]],np.float32)
print(kernel_lap)
lap_img = cv2.filter2D(img, -1, kernel=kernel_lap)
cv2.imshow('lap',lap_img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

# sharpen = ori + edg =》 变清晰了 图象+ edge =更锐利的图象，因为突出边缘
kernel_sharp = np.array([[0,1,0],[1,-3,1],[0,1,0]])
shape_img = cv2.filter2D(img,-1,kernel=kernel_sharp)
# cv2.imshow('img_shape',shape_img)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()

# 这样不对，因为，周围有4个1，中间是-3，虽然有边缘效果，但是周围得1会使得原kernel有滤波效果，使图像模糊；
# 解决：所以取kernel_lap得相反数，再加上原图像，这样突出了中心像素，效果类似于小方差的高斯，所以
#      可以既有边缘效果，又保留图像清晰度
# 二阶导的应用== 》锐化，增强了颗粒感 核里面的值取反 中间值越大，双边效应越大
kernel_sharp2 = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
shape_img2 = cv2.filter2D(img,-1,kernel=kernel_sharp2)
cv2.imshow('img_shape',shape_img2)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

# 更“凶猛”的边缘效果
# 不仅考虑x-y方向上的梯度，同时考虑了对角线方向上的梯度
# edge detection / gradient of x & y axis
edgex = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],np.float32)
shape_img_x = cv2.filter2D(img, -1, kernel=edgex)
# cv2.imshow('sharp_imgx',shape_img_x)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()
edgey = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.float32) # 对上面的转置
shape_img_y = cv2.filter2D(img, -1, kernel=edgex)
cv2.imshow('sharp_imgx',shape_img_x)
cv2.imshow('sharp_imgy',shape_img_y)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
