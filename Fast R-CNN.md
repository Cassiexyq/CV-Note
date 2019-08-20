## Fast R-CNN 

**两大特点**

1. 实现了除了proposal外的end-toend训练，所有特征暂存显存，不需要额外磁盘
2. 提出ROI层

整体流程： 图像输入-》卷积1-》池化1-》。。。-》卷积n-》池化n-》全连接层

直接在feature map上提取所有ROI特征，大大较少了卷积操作，有两个难点

* 如何将原始图像的ROI映射到特征图
* ROI在特征图上的对应的特征区域的维度不满足全连接层的输入怎么办

##### Q：因为全连接层的维度固定死的，所以要使卷积层的输出维度刚好等于全连接层的输入维度？

> * 想办法把不同尺寸的图像也可使池化n产生固定的输出维度。（打破图像输入的固定性）
>
> * 想办法让全连接层可以接收非固定的输入维度
>
>   方法1 是SPPnet的思想，方法2是全连接转成全卷积，作用效果等效于在原始图像做滑窗

**流程：**

​    任意大小的图片输入，经过特征提取层提取特征，特征提取层来自主流分类网络（只能使用到最后一层卷基层）。由selective search等算法生成2000多个region proposal，在特征提取层最后一层进行roi pooling。生成的region proposal是原图的坐标大小，需要映射到特征层，因为原图到最后一个特征层缩小到了原图的1/16，所以将region proposal的坐标乘以1/16就变成了这个region proposal在最后一层特征层的映射坐标。映射到最后一层卷积层后，要经过一个max pooling，这个max pooling是一个简化版的空间金子塔池化。sppnet中的空间金字塔池化是多尺度池化，fast rcnn只使用了一个尺度池化。具体到池化过程，region proposal映射到最后一层特征层是一个矩形区域，将这个映射的矩形区域均分成大小相等的HxW个小矩形，然后在每个小矩形里进行max pooling提取特征。具体到如何分成HxW个小矩形，比如最后一层feature map的大小是60x40，你想分成6x4 = 24个，那每个小矩形的宽度是60/6 = 10，高度是40/4 = 10，之后再按照坐标进行相应计算就好了。 最后HxW个数值进行concatenate成一个特征向量。注意：因为pooling是在每一维的feature map上进行的pooling，所以如果最后一层feature map是256维的，那进行处理后还是生成256维的特征向量，形状就变成了（1，HxW，256）

> 注意点：

1. region proposal是原图的坐标大小，要映射到特征层，每个坐标除以16（VGG）
2. ROI Pooling ,把映射到最后一个特征层的每个ROI都统一到一个size，假如变成3X3，那就将不同尺寸的打格成3×3，然后在每个小格子里进行max pooling就得到3X3的特征map，最后concetrate到一个特征向量
3. 这就涉及到两次量化

 经过roi pooling后，类似于vgg16，接两个全连接层，这两个全连接层都是4096维的。在最后一个全连接层再分别接两个全连接层cls_score和bbox_pred，cls_score的维度是1x种类数，是每个种类的得分，bbox_pred的维度是4x种类数，4代表框的4个变换坐标，即bounding box regression精修。最后再进行loss计算

采用了bounding box regression后，增加了一个bbox_pred层和loss_bbox层。bbox_pred层是一个全连接层，从fc7来，shape是（1,4x种类数）。loss_bbox层（也就是smoothL1）的输入除了bbox_pred外，还有从data层来的bbox_targets，这个就是训练数据的gt框。bbox_pred是由特征层训练得来的，之前一直以为生成的是框的4个坐标（即训练得到最后框的坐标再与gt框进行smoothL1的loss计算），但实际上，bbox_pred是4个变换，即bounding box regression的4个微调。观察不采用bounding box regression的网络结构图，会发现不采用的时候没有bbox_pred，如果bbox_pred表示的是框的4个坐标，那岂不是不使用bounding box regression就不生成框了，这样就不能跑这个模型了，但实际上不使用bounding box regression依旧可以跑模型，所以他代表的是4个变换，即对框坐标的微调。实际上，无论是否使用bounding box regression，最终的框坐标都来自于selective search生成的region proposal，网络的任务是识别这些框是否是某一类东西，如果是某一类东西，那这个框就作为最后检测出结果的框。bounding box regression只不过是对所有这些狂的精修

> 注意点：

1. 可以不采用bbox regression，这样还是会生成框的，只不过错误率变高了
2. cls_score和bbox_pred都由同一层全连接而来，即输入都是特征层，这充分体现了bounding box regression中由特征来学习这4个变换的思想。

测试：

将图片和Rp 输入训练好的网络得到所有的region proposal 的cls_score 和bbox_pred，利用bbox_pred对所有的region proposal进行精修，过滤cls_score小于阈值的，然后nms处理，最后的到结果，如果没有精修，同样做这个处理

1.为什么vgg16最后一层缩小1/16？

vgg16中所有的卷积层都是kenel大小（卷积核大小）为3x3，pad大小为1，stride为1的卷积层。用公式W‘ = (*W − F* + 2*P* )/*S*+1（W代表未经过卷积的feature map大小，F代表卷积核的大小，P代表pad大小，S代表stride大小）计算可以发现，feature map的大小经过卷积后保持不变。vgg16中的卷积层分为5个阶段，每个阶段后都接一个kenel大小为2x2，stride大小为2x2的max pooling，经过一个max pooling后feature map的大小就缩小1/2，经过5次后就缩小1/32。fast rcnn中使用的vgg16只使用第5个max pooling之前的所有层，所以图像大小只缩小1/16。

2.遇到的bug：selective search生成的坐标是（y1，x1，y2，x2），但fast rcnn中的框计算全是（x1，y1，x2，y2）。

3.roi pooling跟faster rcnn的rpn是有区别的，一个是除以16，一个是乘以16。因为一个是从原图映射到特征层，另一个是从特征层映射到原图。