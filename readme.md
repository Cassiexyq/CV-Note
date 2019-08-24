## CV - note

​                                                                                                                                                                                                                                                                                                                          

**1**

* [x] 图象处理：显示，crop，分割三通道，改变颜色通道
* [x] 矩阵运算
* [x] gamma变换，(先归一化待1，用gamma作为指数值求出新的像素值再还原，后查表LUT）
* [x] 直方图，灰度图片均衡化，观察像素分布
* [x] 旋转：similarity transform，Affine Transform，Perspective Transform三者的区别
* [x] 沿横轴放大加平移，沿x轴剪切变换，旋转，顺时针旋转，多种组合

**2**

​	参考：<https://blog.csdn.net/wsp_1138886114/article/details/81368890#13_Sobel_30>

* [x] 图象卷积： 卷积翻转的有无必要性问题
* [x] 图像梯度变化，一阶导核二阶导的简单理解
* [x] 二阶导的双边效应核精细结构，增强作用
* [x] 高斯模糊，能够加速的原因，如何做到模糊
* [x] 二阶导的应用=》锐化，增加了颗粒感，核取反，类似小方差的高斯，保持了图像清晰度同时有边缘效果；
* [x] sobel算子（一阶导）：可以考虑x-Y方向的梯度，也考虑对角线的梯度
* [x] Harris角点
* [x] SIFT 详解

**3**

* [x] 逻辑回归推公式+手撸
* [x] 线性回归 推公式+手撸
* [x] 线性回归和逻辑回归的不同（从代码角度来讲的话）： 交叉熵（损失函数不一样，假设函数不一样，一个w,b，一个theta，w,b是一个值，theta是一个矩阵，维度跟特征数有关）
* [x] 监督（分类和回归）-非监督， 

**4**

- [x] 反向传播，两层CNN手撸代码
- [x] L1VSL2正则   L1loss vs L2loss
- [x] SVM详解

**5：coding**

Quick sort/ Top k

DFS/BFS/并查集

求幂次，平方根，求极值

**6**

- [x] CNN卷积方式
- [x] relu的非线性
- [x] 池化： 最大池化和平均池化
- [x] dropout  训练和测试注意的问题， 优点
- [x] BN层，如何使用

**7**

- [ ] 网络初始化 Gussian/ Xavier / Kaiming
- [ ] PCA
- [ ] 优化方法 SGD，SGD+Momentum，Nesterov，Adagrad，Adam

**8**

- [ ] VGG
- [ ] GoogLeNet  1×1卷积的效果 auxiliaty softmax 不同卷积核的结合类似图像金字塔
- [ ] ResNet elementwise Addition  shortcut=bottleneck
- [ ] mobileNet-v1: Depthwise+Printwise  v2: Inverted residual+ linear Bottlenecks
- [ ] ShuffleNet
- [ ] FLOPS的计算

**9**

- [ ] Softmax 交叉熵梯度计算
- [ ] Caffe 使用简单介绍
- [ ] pytorch 使用简单介绍
- [ ] 如何解决数据不平衡，从数据，从Loss，从学习策略，Center Loss，Attention机制

**10** [参考](<https://imlogm.github.io/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/rcnn/>)

- [ ] Two Stage Detection: RCNN Fast RCNN  Faster RCNN
- [ ] NMS 和 Soft-NMS
- [ ] One Stage Detection: Yolo  RetainNet



