# -*- coding: utf-8 -*-

# @Author: xyq

import cv2
import numpy as np


'''
高斯分布， var 取大取小
 var 大：图变得越模糊
 var 小： 峰值更趋向于原点，周围的值对中间值的影响更小，所以图会变得比较清晰
'''
img = cv2.imread('th.jpg')
# cv2.imshow('beauty',img)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()
g_img = cv2.GaussianBlur(img,(7,7),5)  #  (7，7)表示取值范围， 5 表示方差
# cv2.imshow('guu',g_img)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()
g_img2= cv2.GaussianBlur(img,(7,7),0.1)
# cv2.imshow('guu',g_img2)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()

kernel = cv2.getGaussianKernel(7,5)  # 7:size 5:variance
print(kernel)
# [[0.12895603]
#  [0.14251846]
#  [0.15133131]
#  [0.1543884 ]
#  [0.15133131]
#  [0.14251846]
#  [0.12895603]]
# 这里返回只有一个 vector ，另一个是这个的转置，两个值是一样的，计算的时候有两个这样的vector
# 为什么高斯计算的时候要变成先按一个方向卷积，再算另一个方向卷积=》高斯能够加速的原因 运算次数减少

g_1 = cv2.GaussianBlur(img, (7,7),5)  # 隐式求卷积
g_2 = cv2.sepFilter2D(img, -1, kernel,kernel) # 显示求卷积 ，两个结果是一样的
# cv2.imshow('gu1',g_1)
# cv2.imshow('gu2',g_2)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()

