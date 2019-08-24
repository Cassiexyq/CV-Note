# YOLO

### yolo1

作者在YOLO算法中把物体检测（object detection）问题处理成回归问题，用一个卷积神经网络结构就可以从输入图像直接预测bounding box和类别概率。

算法首先把输入图像划分成S*S的格子，然后对每个格子都预测B个bounding boxes，每个bounding box都包含5个预测值：x,y,w,h和confidence。x,y就是bounding box的中心坐标，与grid cell对齐（即相对于当前grid cell的偏移值），使得范围变成0到1；w和h进行归一化（分别除以图像的w和h，这样最后的w和h就在0到1范围）。
每个bounding box都对应一个confidence score，如果grid cell里面没有object，confidence就是0，如果有，则confidence score等于预测的box和ground truth的IOU值

**所以如何判断一个grid cell中是否包含object呢**？答案是：如果一个object的ground truth的中心点坐标在一个grid cell中，那么这个grid cell就是包含这个object，也就是说这个object的预测就由该grid cell负责。 

具体判断boundingbox 内预测类别的乘法操作：

即得到每个bounding box属于哪一类的confidence score。也就是说最后会得到20x（7x7x2）=20x98的score矩阵，括号里面是bounding box的数量，20代表类别。接下来的操作都是**20个类别轮流进**行：在某个类别中（即矩阵的某一行），将得分少于阈值（0.2）的设置为0，然后再按得分从高到低排序。最后再用NMS算法去掉重复率较大的bounding box。最后每个bounding box的20个score取最大的score，如果这个score大于0，那么这个bounding box就是这个socre对应的类别（矩阵的行），如果小于0，说明这个bounding box里面没有物体，跳过即可。

（NMS:针对某一类别，选择得分最大的bounding box，然后计算它和其它bounding box的IOU值，如果IOU大于0.5，说明重复率较大，该得分设为0，如果不大于0.5，则不改；这样一轮后，再选择剩下的score里面最大的那个bounding box，然后计算该bounding box和其它bounding box的IOU，重复以上过程直到最后）

网络方面主要采用GoogLeNet，卷积层主要用来提取特征，全连接层主要用来预测类别概率和坐标。最后的输出是7x7x30，这个30前面也解释过了，7x7是grid cell的数量。这里注意下实现的细节可能人人都不大一样，比如对inception的改动，最后几层的全连接层的改动等等，但是重点在于最后一层的输出是7x7x30。

yolo v1 的价值

1. 作者先在ImageNet数据集上预训练网络，而且网络只采用fig3的前面20个卷积层，输入是224x224大小的图像。然后在检测的时候再加上随机初始化的4个卷积层和2个全连接层，同时输入改为更高分辨率的448x448

2. Relu层改为pRelu，即当x<0时，激活值是0.1*x，而不是传统的0。

3. 作者采用sum-squared error的方式把localization error（bounding box的坐标误差）和classificaton error整合在一起。但是如果二者的权值一致，容易导致模型不稳定，训练发散。因为很多grid cell是不包含物体的，这样的话很多grid cell的confidence score为0。所以采用设置不同权重方式来解决，一方面提高localization error的权重，另一方面降低没有object的box的confidence loss权值，loss权重分别是5和0.5。而对于包含object的box的confidence loss权值还是原来的1

   这里注意用宽和高的开根号代替原来的宽和高，这样做主要是因为相同的宽和高误差对于小的目标精度影响比大的目标要大。举个例子，原来w=10，h=20，预测出来w=8，h=22，跟原来w=3，h=5，预测出来w=1，h=7相比，其实前者的误差要比后者小，但是如果不加开根号，那么损失都是一样：4+4=8，但是加上根号后，变成0.15和0.7。 

**所以具体实现的时候是什么样的过程呢？**

训练的时候：输入N个图像，每个图像包含M个object，每个object包含4个坐标（x，y，w，h）和1个label。然后通过网络得到7*7*30大小的三维矩阵。每个1x30的向量前5个元素表示第一个bounding box的4个坐标和1个confidence，第6到10元素表示第二个bounding box的4个坐标和1个confidence。最后20个表示这个grid cell所属类别。注意这30个都是预测的结果。然后就可以计算损失函数的第一、二 、五行。至于第二三行，confidence可以根据ground truth和预测的bounding box计算出的IOU和是否有object的0,1值相乘得到。真实的confidence是0或1值，即有object则为1，没有object则为0。 这样就能计算出loss function的值了。

1）预训练。使用 ImageNet 1000 类数据训练YOLO网络的前20个卷积层+1个average池化层+1个全连接层。训练图像分辨率resize到224x224。

2）用步骤1）得到的前20个卷积层网络参数来初始化YOLO模型前20个卷积层的网络参数，然后用 VOC 20 类标注数据进行YOLO模型训练。**检测通常需要有细密纹理的视觉信息,所以为提高图像精度，**在训练检测模型时，将输入图像分辨率从224 × 224 resize到448x448。

训练时B个bbox的ground truth设置成相同的.

测试的时候：输入一张图像，跑到网络的末端得到7x7x30的三维矩阵，这里虽然没有计算IOU，但是由训练好的权重已经直接计算出了bounding box的confidence。然后再跟预测的类别概率相乘就得到每个bounding box属于哪一类的概率。
**缺点**： 

1. 占比较小的目标检测效果不好.虽然每个格子可以预测B个bounding box，但是最终只选择只选择IOU最高的bounding box作为物体检测输出，即每个格子最多只预测出一个物体。当物体占画面比例较小，如图像中包含畜群或鸟群时，每个格子包含多个物体，但却只能检测出其中一个。
2. 输入尺寸固定：由于输出层为全连接层，因此在检测时，YOLO训练模型只支持与训练图像相同的输入分辨率。其它分辨率需要缩放成改分辨率.
3. YOLO与Fast R-CNN相比有较大的定位误差，与基于region proposal的方法相比具有较低的召回率

### yolo v2 

 **1.batch normalization**： BN能够给模型收敛带来显著地提升，同时也消除了其他形式正则化的必要。作者在每层卷积层的后面加入BN后，在mAP上提升了2%。BN也有助于正则化模型。有了BN便可以去掉用dropout来避免模型过拟合的操作。BN层的添加直接将mAP硬拔了2个百分点，这一操作在yolo_v3上依然有所保留，BN层从v2开始便成了yolo算法的标配。

 **2.high resolution classifier**：所有最顶尖的检测算法都使用了基于ImageNet预训练的分类器。从AlexNet开始，大多数分类器的输入尺寸都是小于256x256的。最早的YOLO算法用的是224x224，现在已经提升到448了。这意味着网络学习目标检测的时候必须调整到新的分辨率。  发现mAP提升了4% 。

> **原来的YOLO网络在预训练的时候采用的是224\*224的输入（这是因为一般预训练的分类模型都是在ImageNet数据集上进行的），然后在detection的时候采用448\*448的输入**，这会导致从分类模型切换到检测模型的时候，模型还要适应图像分辨率的改变。

1. 在 ImageNet 数据训练 224*224,**大概160个epoch**
2. **输入调整到448\*448，再训练10个epoch,也是ImageNet**
3. 又在自己的数据集FT，得到 13X13的feature map

**3. Convolutional With Anchor Boxes**: 在yolo_v2的优化尝试中加入了anchor机制。YOLO通过全连接层直接预测Bounding Box的坐标值。首先将原网络的**全连接层和最后一个poolin**g层去掉，使得最后的卷积层可以有更高分辨率的特征；然后缩减网络，用416x416大小的输入代替原来448x448。这样做的原因在于希望得到的特征图都有**奇数**大小的宽和高，奇数大小的宽和高会使得每个特征图在划分cell的时候就只有一个center cell（比如可以划分成7x7或9x9个cell，center cell只有一个，如果划分成8x8或10x10的，center cell就有4个）

> **为什么希望只有一个center cell呢？因为大的object一般会占据图像的中心，所以希望用一个center cell去预测，而不是4个center cell去预测。网络最终将416\*416的输入变成13\*13大小的feature map输出，也就是缩小比例为32。** 

v1中直接在卷积层之后使用全连接层预测bbox的坐标。v2借鉴Faster R-CNN的思想预测bbox的偏移

**移除了全连接层,并且删掉了一个pooling层**使特征的分辨率更大一些.另外调整了网络的输入(448->416)以使得位置坐标是奇数只有一个中心点(yolo使用pooling来下采样,有5个size=2,stride=2的max pooling,而卷积层没有降低大小,因此最后的特征是416/(2^5)=13).v1中每张图片预测7x7x2=98个box,而v2加上Anchor Boxes能预测超过1000个.检测结果从69.5mAP,81% recall变为69.2 mAP,88% recall.

Faster R-CNN并不是直接预测坐标值。Faster R-CNN只是用RPN的全连接来为每一个box预测offset（坐标的偏移量或精修量）以及置信度（得分）,由于预测层是卷积性的，所以RPN预测offset是全局性的。预测offset而不是坐标简化了实际问题，并且更便于网络学习。

> **（说明：faster r-cnn的box主体来自anchor，RPN只是提供精修anchor的offset量）**

我们知道原来的YOLO算法将输入图像分成7x7的网格，每个网格预测两个bounding box，因此一共只有98个box，但是在YOLOv2通过引入anchor boxes，预测的box数量超过了1千（以输出feature map大小为13x13为例，每个grid cell有9个anchor box的话，一共就是13x13x9=1521个，**当然由后面第4点可知，最终每个grid cell选择5个anchor box**）。顺便提一下在Faster RCNN在输入大小为1000x600时的boxes数量大概是6000，在SSD300中boxes数量是8732。显然增加box数量是为了提高object的定位准确率。 
作者的实验证明：虽然加入anchor使得MAP值下降了一点（69.5降到69.2），但是提高了recall（81%提高到88%）。

**4. Dimension Clusters**: 当作者对yolo使用anchor机制时，遇到了两个问题。

​	1，Faster R-CNN中**k=9**，大小尺寸一共有3x3种，anchor的长宽比是跟为人为经验设置的。box的规格虽然后期可以通过线性回归来调整，**但如果一开始就选用更合适的anchor box的话，可以使网络学习更轻松一些。**
作者并没有手动设定prior，而是在训练集的b-box上用了k-means聚类来自动找到prior。**如果用标准k-means(使用欧几里得距离)，在box的尺寸比较大的时候其误差也更大**。然而，我们真正想要的是能够使IOU得分更高的优选项，与box的大小没有关系。因此，对于距离判断，作者用了： 
d(box, centroid) = 1 - IOU(box, centroid) 
最终选择了**k=5**，这是在模型复杂度和高召回率之间取了一个折中。聚类得到的框和之前手动挑选的框大不一样。有稍微短宽的和高瘦一些的(框)。

​	2.模型不稳定。不稳定的因素主要来自于为box预测(x,y)位置的时候。在RPN中，网络预测了值tx和ty以及(x, y)坐标。直接预测(x, y)，就像yolo_v1的做法，不过v2是**预测一个相对位置，相对单元格的左上角的坐标**，当(x, y)被直接预测出来，那整个bounding box还差w和h需要确定。yolo_v2的做法是既有保守又有激进，x和y直接暴力预测，而w和h通过回归的调整来确定

cx和cy表示一个cell和图像左上角的横纵距离，中心点（tx和ty）经过sigmoid函数处理后范围在0到1之间,归一化处理也使得模型训练更加稳定

$$b_x = \sigma(t_x) + c_x​$$

$$b_y = \sigma(t_y) + c_y$$

$$b_w = p_we^{t_w}​$$

$$b_h = p_he^{t_h}$$

**在这里作者并没有采用直接预测offset的方法，还是沿用了YOLO算法中直接预测相对于grid cell的坐标位置的方式**

**5. fine-Grained Features**

这里主要是添加了一个层：passthrough layer。这个层的作用就是将前面一层的26x26的feature map和本层的13x13的feature map进行连接，有点像ResNet。这样做的原因在于虽然13x13的feature map对于预测大的object以及足够了，但是对于预测小的object就不一定有效。也容易理解，越小的object，经过层层卷积和pooling，可能到最后都不见了，所以通过合并前一层的size大一点的feature map，可以有效检测小的object。

**6. multi scaling Training**

简单讲就是在训练时输入图像的size是动态变化的，**注意这一步是在检测数据集上fine tune时候采用的，不要跟前面在Imagenet数据集上的两步预训练分类模型混淆**,在训练网络时，每训练10个epoch，网络就会随机选择另一种size的输入。那么输入图像的size的变化范围要怎么定呢？前面我们知道本文网络本来的输入是416x416，最后会输出13x13的feature map，也就是说downsample的factor是32，因此作者采用32的倍数作为输入的size，具体来讲文中作者采用从{320,352,…,608}的输入尺寸。 
**这种网络训练方式使得相同网络可以对不同分辨率的图像做detection**



1、Training for Classification 
这里的2和3部分在前面有提到，就是训练处理的小trick。这里的training for classification都是在ImageNet上进行预训练，主要分两步：1、从头开始训练Darknet-19，数据集是ImageNet，训练160个epoch，输入图像的大小是224*224，初始学习率为0.1。另外在训练的时候采用了标准的数据增加方式比如随机裁剪，旋转以及色度，亮度的调整等。2、再fine-tuning 网络，这时候采用448x448的输入，参数的除了epoch和learning rate改变外，其他都没变，这里learning rate改为0.001，并训练10个epoch。结果表明fine-tuning后的top-1准确率为76.5%，top-5准确率为93.3%，而如果按照原来的训练方式，Darknet-19的top-1准确率是72.9%，top-5准确率为91.2%。因此可以看出第1,2两步分别从网络结构和训练方式两方面入手提高了主网络的分类准确率。

2、Training for Detection 
在前面第2步之后，就开始把网络移植到detection，并开始基于检测的数据再进行fine-tuning。首先把最后一个卷积层去掉，然后添加3个3x3的卷积层，每个卷积层有1024个filter，而且每个后面都连接一个1x1的卷积层，1x1卷积的filter个数根据需要检测的类来定。比如对于VOC数据，由于每个grid cell我们需要预测5个box，每个box有5个坐标值和20个类别值，所以每个grid cell有125个filter（与YOLOv1不同，在YOLOv1中每个grid cell有30个filter，还记得那个7x7x30的矩阵吗，**而且在YOLOv1中，类别概率是由grid cell来预测的**，也就是说一个grid cell对应的两个box的类别概率是一样的，但是在YOLOv2中，类别概率是属于box的，每个box对应一个类别概率，而不是由grid cell决定，因此这边每个box对应25个预测值（5个坐标加20个类别值），而在YOLOv1中一个grid cell的两个box的20个类别值是一样的）。另外作者还提到将最后一个3x3x512的卷积层和倒数第二个卷积层相连。最后作者在检测数据集上fine tune这个预训练模型160个epoch，学习率采用0.001，并且在第60和90epoch的时候将学习率除以10，weight decay采用0.0005。

> 因为YOLOv1是 20+5+5=30个fiter，类别概率是有cell来预测的，也就是说2个bbox在20类别上的预测值是一样的；而YOLOv2是 5个box ，每个box有20+5要预测，有125filter，类别概率是box来预测的

引入一点：**YOLO，YOLOv2、YOLO9000，Darknet-19，Darknet-53，YOLOv3 分别是什么关系？**

1. YOLOv2 是 YOLO 的升级版，但并不是通过对原始加深或加宽网络达到效果提升，反而是简化了网络。
2. YOLO9000 是 CVPR2017 的最佳论文提名。首先讲一下这篇文章一共介绍了 YOLOv2 和 YOLO9000 两个模型，二者略有不同。前者主要是 YOLO 的升级版，后者的主要检测网络也是 YOLOv2，同时对数据集做了融合，使得模型可以检测 9000 多类物体。而提出 YOLO9000 的原因主要是目前检测的数据集数据量较小，因此利用数量较大的分类数据集来帮助训练检测模型。
3. YOLOv2 使用了一个新的分类网络作为特征提取部分，参考了前人的先进经验，比如类似于 VGG，作者使用了较多的 3 * 3 卷积核，在每一次池化操作后把通道数翻倍。借鉴了 network in network 的思想，网络使用了全局平均池化（global average pooling），把 1 * 1 的卷积核置于 3 * 3 的卷积核之间，用来压缩特征。也用了 batch normalization（前面介绍过）稳定模型训练。最终得出的基础模型就是 Darknet-19，如上图，其包含 19 个卷积层、5 个最大值池化层（maxpooling layers ）

### YOLO V3

改进

1. 多尺度（类FPN）
2. 更好的基础分类网络（类RESNET）和分类器

整个v3结构里面，是**没有池化层和全连接层**的。前向传播过程中，**张量的尺寸变换是通过改变卷积核的步长来实现的**，不同于faster R-CNN的是，yolo_v3只会对1个prior进行操作，也就是那个最佳prior。而logistic回归就是用来从9个anchor priors中找到objectness score(目标存在可能性得分)最高的那一个。logistic回归就是用曲线对prior相对于 objectness score映射关系的线性建模。


第一点， 9个anchor会被三个输出张量平分的。根据大中小三种size各自取自己的anchor。

第二点，每个输出y在每个自己的网格都会输出3个预测框，这3个框是9除以3得到的，这是作者设置
的，我们可以从输出张量的维度来看，13x13x255。255是怎么来的呢，3x(5+80)。80表示80个种类，5表
示位置信息和置信度，3表示要输出3个prediction。在代码上来看，3x(5+80)中的3是直接由
num_anchors//3得到的。

第三点，作者使用了**logistic回归**来对每个anchor包围的内容进行了一个目标性评分(objectness score)。
根据目标性评分来选择anchor prior进行predict，而不是所有anchor prior都会有输出。



YOLOV2<https://blog.csdn.net/u014380165/article/details/77961414>

​	< https://blog.csdn.net/leviopku/article/details/82588959>

YOLO<https://www.cnblogs.com/makefile/p/YOLOv3.html>

YOLOv3<https://blog.csdn.net/leviopku/article/details/82660381>