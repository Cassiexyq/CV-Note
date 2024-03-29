# Faster R-CNN

* 改进
  * RPN 替代SS生成region proposal （开创了利用卷积网络、特征层来生成region proposal，做到了share computation，提高了运行效率）
  * 实现了真正意义上的端到端（fast rcnn有selective search单独生成region proposal的过程，不算真正意义上的端到端）

<https://www.cnblogs.com/ymjyqsx/p/7661088.html>

faster rcnn分成rpn网络和fast rcnn网络，fast rcnn网络和之前讲的一样，只是region proposal部分不来自selective search，而来自rpn网络（实际上就是一个rpn网络结合在fast rcnn网络上）。rpn网络和fast rcnn网络共用特征提取层，体现了share computation的思想。单独来看rpn网络，无论特征提取层用什么网络，都是在特征提取层最后一层卷积后面先添加一个kenel大小为3x3，stride为1，pad为1的卷积层，经过卷积后，feature map大小与特征层最后一层的feature map保持不变。

之后再分别接一个属于rpn_cls_score的卷积，和一个属于rpn_bbox_pred的卷积。这两个卷积都是1x1的卷积，不同的是，rpn_cls_score的维度为2x种类数，rpn_bbox_pred的维度为4x种类数（可以看到这里利用了1x1卷积的降维功能）。实际上rpn_cls_score就是某一类的框为前景、背景的预测概率值，rpn_bbox_pred就是每一类的框的预测坐标值。值得注意的是：这里无论是score还是bbox坐标，都是直接从上一层卷积再卷积而来的，也就是从特征映射而来的，这也体现了从特征做判断的思想，在整个模式识别领域，实质上都是通过特征去判断去识别。

更值得注意的是：这里映射得到的bbox坐标并不是直接的框的四个坐标值，而是四个变化值，即bounding box regression，所以，rpn网络训练学习的并不是直接的四个坐标值，而是4个变化值，这4个变换值，也就是rpn_bbox_pred，会输出到smoothL1。这与fast rcnn网络中使用bounding box regression很类似。不同的是，在fast rcnn网络中，smoothL1计算的是4个变化值和gt框的loss，但在rpn网络smoothL1计算的是**4个变换值和4个变换值的loss**。第一个4个变换值是从网络特征层提取的，第二个4个变换值是anchor和gt框的之间的变换值。第二个4个变换值是由rpn-data层来实现的（代码是rpn.anchor_target_layer）。rpn-data层输出的是gt框和anchor之间的4个变换值，也就是如何让anchor更加接近gt框，loss计算的就是这两个变换值的loss。

在下一个小阶段，直接由anchors根据网络训练的4个变换值在生成最终的anchor坐标，这样也就接近原始数据中的gt框。rpn网络训练过程中，要筛选出256个anchor作为loss计算，正例128个，负例128个。

有两种anchor为正例：

​	1.anchor是所有anchor中与某一个gt的iou最大　　

​	2.anchor只要和一个gt的iou大于0.7。

只有一种anchor为负例：

与所有的gt的iou都小于0.3。

注意：虽然两种情况都是positive，但anchor和gt计算4个变换时计算的是anchor和与这个anchor有最大iou的gt框的的4个变换

源码

<https://github.com/tryolabs/luminoth/blob/master/luminoth/models/fasterrcnn/fasterrcnn.py>