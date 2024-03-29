#### 图像的卷积

* 一阶导  图像上的梯度变化

  $$f'(x) = f(x+1) - f(x)$$

* 二阶导

  $$f''(x) = f'(x+1) - f'(x) = f(x+2)- f(x+1) - f(x+1) + f(x) = f(x+1) + f(x-1) - 2f(x)$$

> #### 图像卷积核有没有必要翻转 -- 没有必要
>
> 首先对于传统CV来说，绝大部分的卷积核如高斯核都是对称的，对它翻转不翻转都是一样的卷积核
>
> 对于深度的CNN来说，像素值对模型而言是未知的，这是要通过训练，经过BP算法等自动算出来的，核转于不转，它都能自己算出来，没有影响

>  对*一*阶导的应用：
>
> **sobel算子**： 可以对水平，垂直，对角线方向检测里面就涉及到一阶隔一个算，这样更为稳定

> 对二阶导的应用：一阶一般能产生比较粗糙的边缘，二阶就有了增强的作用，双边效应，精细结构，对突兀点有增强突出的作用
>
> **Lapacian算子**：锐化图片，增加图像颗粒感，核的值取反，中间值变为最大，类似一个小方差的高斯核，保持了图像清晰度又有边缘效果。
>
> 高斯拉普拉斯就是先高斯函数求二阶导再与原图卷积。

区别：一阶一般用于检测边缘，线的检测，二阶就用于特征点的检测

共同点：求导的涉及问题用来求角点，边缘，比较“尖”的东西

#### 高斯模糊：

是指中间的核数很大，周围很小，这样就会有中间的值被周围小数平均掉，就起到了平滑作用。

​		$$h(x,y) = \frac{1}{2\pi\sigma^2}e^{\frac{-(x^2+y^2)}{2\sigma^2}}$$

> #### 为什么高斯有加速作用
>
> 普通的卷积操作都是拿一整个卷积核去做卷积（3*3的卷积核），高斯不是这样，先一个方向做卷积，再另一个方向做卷积，这样的好处就是，原本做一次二维操作的卷积，就有9次乘法+8次加法，换为两次卷积，就变成2×(3次乘法+2次加法)，这就起到了加速作用

#### Harris角点

做这个角点需要把图像转灰度才可以，具体推到看博客，主要结论就是这个角点是通过计算超过需自己设定的输入的阈值设为特征点，**它不能满足尺度不变性的角点，但满足旋转不变性的角点，对亮度影响变化的角点 **也满足（k 一般设为0.05）

$$\lambda_1 \lambda_2 - k(\lambda_1+\lambda_2) >= thresh  $$    

#### SIFT

1. 利用高斯核生成一组（octave）的的6层尺寸相同但模糊系数不同的图像层（**高斯核是实现尺度不变的唯一线性核**），因为有尺度，所以会对图像做一个resize，然后又生成一组的图像层。就会有很多层。

2. 因为sift 满足尺度不变性，目的是为了在各个尺度下都不会变的角点，所以要模拟各个尺度-》图像金字塔，说明如果在各个尺度内都有这个点说明这个是一个好的特征点。

   寻找关键点

   3. 相邻的图做差，求取梯度，能够得到边缘图像，就可以得到不同尺度下的边缘图像-》就是高斯差分金字塔的尺度空间，**（这里从一开始就需要建立高斯差分金字塔，顺序是为了解释）**
   4. 找到极值点，去找不同尺度的极值点，即不仅要考虑图像本身还要考虑相邻空间（不同尺度的邻居）图像的极值点作为**候选点**，这个点和它周围26个点共27个点取值最大和最小加入候选
   5. 获得更为精准的关键点，首先，这些点是离散的点，拟合曲线求导获得极值，可以去掉一些非极值点
   6. 抑制噪声，去除敏感点
   7. 对每个关键点赋予方向，方向是算出来的，利用梯度求角度，对角度根据360度取划分数，得每个范围加权获得一个直方图，峰值就是特征点的主方向，如果次峰值是峰值的80%以上，可以作为特征点的第二主方向，之后会有8个方向给予这个关键点
   8. 因为SIFT还满足旋转不变性，可以变成没有角度，即角度对它没有影响，最后，以关键点为中心，做一个4×4的cell，然后得到每个cell的直方图，每个直方图有8个方向，所以一共有4×4×8=128个方向向量，叫做特征描述子，descriptor

   其中，高斯差分金字塔，上一层和下一层的关系，后生成的组的第一张图是由上一层倒数第三个直接降采样不需模糊产生，同一组又是由第一张基础上使用连续的高斯系数产生，不需要降采样。

