# SSD

这种算法对于不同横纵比的object的检测都有效，这是因为算法对于每个feature map cell都使用多种横纵比的default boxes，这也是本文算法的核心。另外本文的default box做法是很类似Faster RCNN中的anchor的做法的。最后本文也强调了增加数据集的作用，包括随机裁剪，旋转，对比度调整等等。 


**文中作者提到该算法对于小的object的detection比大的object要差。作者认为原因在于这些小的object在网络的顶层所占的信息量太少，所以增加输入图像的尺寸对于小的object的检测有帮助**

**另外增加数据集对于小的object的检测也有帮助，原因在于随机裁剪后的图像相当于“放大”原图像，所以这样的裁剪操作不仅增加了图像数量，也放大了图像。**

### 概述：

**提出的SSD算法是一种直接预测bounding box的坐标和类别的object detection算法**，算法的主网络结构是VGG16，将两个全连接层改成卷积层再增加4个卷积层构造网络结构。对其中5个不同的卷积层的输出分别用两个3*3的卷积核进行卷积，一个输出分类用的confidence，每个default box生成21个confidence（这是针对VOC数据集包含20个object类别而言的）；一个输出回归用的localization，每个default box生成4个坐标值（x，y，w，h）。另外这5个卷积层还经过priorBox层生成default box（生成的是坐标）
**default box，是指在feature map的每个小格(cell)上都有一系列固定大小的box**

假设每个feature map cell有$k$个default box，那么对于每个default box都需要预测c个类别score和4个offset，那么如果一个feature map的大小是mxn，也就是有mxn个feature map cell，那么这个feature map就一共有$k*m*n$个default box，每个default box需要预测4个坐标相关的值和c+1个类别概率


> （实际code是分别用不同数量的3x3卷积核对该层feature map进行卷积，比如卷积核数量为（c+1）x k对应confidence输出，表示每个default box的confidence，就是类别；卷积核数量为 4xk 对应localization输出，表示每个default box的坐标）

所以这里用到的default box和Faster RCNN中的anchor很像，在Faster RCNN中anchor只用在最后一个卷积层，但是在本文中，default box是应用在多个不同层的feature map上。

**下图还有一个重要的信息是：在训练阶段，算法在一开始会先将这些default box和ground truth box进行匹配，比如蓝色的两个虚线框和猫的ground truth box匹配上了，一个红色的虚线框和狗的ground truth box匹配上了。所以一个ground truth可能对应多个default box。在预测阶段，直接预测每个default box的偏移以及对每个类别相应的得分，最后通过NMS得到最终的结果**

那么default box的scale（大小）和aspect ratio（横纵比）要怎么定呢？假设我们用m个feature maps做预测，那么对于每个featuer map而言其default box的scale是按以下公式计算的

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170531223831240?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

这里smin是0.2，表示最底层的scale是0.2,；smax是0.9，表示最高层的scale是0.9。 
至于aspect ratio，用ar表示为下式：注意这里一共有5种aspect ratio 

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170531223908725?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

因此每个default box的宽的计算公式为

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170531223919225?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

高的计算公式为：（很容易理解宽和高的乘积是scale的平方）

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170531223928678?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

另外当aspect ratio为1时，作者还增加一种scale的default box： 

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170531223938350?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

因此，对于每个feature map cell而言，一共有6种default box。 

**可以看出这种default box在不同的feature层有不同的scale，在同一个feature层又有不同的aspect ratio，因此基本上可以覆盖输入图像中的各种形状和大小的object！**

显然，当default box和grount truth匹配上了，那么这个default box就是positive example（正样本），如果匹配不上，就是negative example（负样本），显然这样产生的负样本的数量要远远多于正样本。于是作者将负样本按照confidence loss进行排序，然后选择排名靠前的一些负样本作为训练，使得最后负样本和正样本的比例在3:1左右。

**损失函数方面**：和Faster RCNN的基本一样，由分类和回归两部分组成，可以参考Faster RCNN，这里不细讲。总之，回归部分的loss是希望预测的box和default box的差距尽可能跟ground truth和default box的差距接近，这样预测的box就能尽量和ground truth一样。

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170531224728103?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

