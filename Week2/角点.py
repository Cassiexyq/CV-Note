# -*- coding: utf-8 -*-

# @Author: xyq


import cv2
import numpy as np

img = cv2.imread('th.jpg')

'''
harris 角点
'''
img_gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
img_harris = cv2.cornerHarris(img_gray, 2, 3, 0.05)
print(img_harris)
# cv2.imshow("img_harris", img_harris)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()
# 没法看原因：1. float类型； 2. img_harris本质上是每个pixel对于Harris函数的响应值
# 没有看的价值
# 为了显示清楚
img_harris = cv2.dilate(img_harris, None)  # 膨胀操作,变大
thres = 0.05 * np.max(img_harris)
print(thres)
img[img_harris > thres] = [0,0,255]
cv2.imshow("img_haris", img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

'''
SIFT
'''

# create sift class

sift = cv2.xfeatures2d_SIFT()
# detect SIFT
kp = sift.detect(img,None)   # None for mask
# compute SIFT descriptor
kp,des = sift.compute(img,kp)
print(des.shape)
img_sift= cv2.drawKeypoints(img,kp,outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('lenna_sift.jpg', img_sift)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()