### RCNN

> 1. Region Proposal  ： 获得候选框，大约2K个
> 2. Feature Extraction : 提取特征
> 3. SVM分类+ NMS + BBox 回归

[主要特定]

* 速度： 预先提取一系列可能是物体的候选区域，之后仅在候选区域熵提取特征
* 训练集：使用深度网络进行特征提取，使用了两个数据集，ImageNet和PASCAL VOC ，前者是识别库，后者是检测库，本文使用识别库进行预训练得到CNN，再使用检测库进行调优，最后在检测库上评测。

**流程**

* 每个输入的图片利用SS获得候选框生成2k的候选区域，可能是物体的区域，有大有小

  > 正负样本，用ss挑选出来的候选框与物体的工人标注矩形框的重叠区域IOU大于0.5为正样本

* 对每个区域，送入CNN前，需要同一尺寸，把每个候选框送入CNN提取特征

* **每张图的候选区域都要送入一个CNN，是每张图，而且送入CNN是候选区域要同一size的**

* 得到的是feature vector，然后送入SVM，判断是否属于该类

* 使用回归其精细修正候选框

**特征提取步骤中**

1. 选择的batchsize： 128 = 32pos+96neg（background）[iou<0.5] *32个样本和96个背景*
2. 利用的是预训练网络
3. 用现有数据对网络finetune，如果不针对特定任务进行finetune，卷积层学到的其实就是基础的共享特征提取鞥，用于提取各种图片的特征，而fc6 fc7全连接所学到的特征是用于针对特定任务的特征。

**第三步**

得到固定长度的4096维的feature vector后需要分类

在SVM中的iou和训练CNN的iou不一样，因为训练CNN需要更多的数据

类别判断，因为负样本较多，使用hard nefitive mining方法。（iou<0.3）

**nms vs soft-nms**

传统的NMS是没有考虑score值的，只考虑了iou值，实现的原理就是： 对score进行排序，对当前最大的score的bbox拿出来跟剩下的bbox进行iou对比，如果iou>设定的阈值，就扔掉，然后再从剩下的score中选择最大的，再于剩下的bbox进行iou对比，扔掉iou大的，直到bbox都比了一遍。

soft-nms把两个点都考虑了进去，不仅是score值，还有iou值，所以每次拿出score的bbox跟剩下的bbbox进行比较，对score重新计算f(iou)，去掉得分最低的。计算的函数是跟iou成反比例关系，当iou越大，影响越小，如果当前iou值小于阈值，保留这个score值，如果大于了阈值，重新计算score，有两种加权操作来计算，一种是线性加权（这是一个不连续的函数问题），拿score直接乘1-iou值；另一种高斯加权（连续），这样的话，iou越高，si值越低，iou越低，si值越高，iou为0，两个框没有交叠，两个仍然需要保留。

### Q：候选区域提出阶段所产生的结果尺寸不同？

> RCNN提取阶段采用的是AlexNet，最后两层是全连接层fc6和fc7，所以必须保证输入的图片尺寸相同。而候选区域的产生结果是不相同的，RCNN采用多种方式对图片进行放缩，最后发现是各向异性加padding最好。

### Q: CNN训练本来就是对bounding box物体进行识别，最后加一层softmax就是分类层，为什么还要先用CNN进行特征提取（fc7），再用于训练SVM

> 这是因为SVM训练和CNN训练过程的正负样本定义方式不同，导致最后采用的CNN softmax的输出比采用SVM的精度还低。CNN在训练的时候，对训练的数据采用比较宽松的标注，如果bbox包含物体的一部分也标注成了正样本，采用这种方法的原因是CNN容易过拟合，需要大量训练数据，所以设置条件比较宽松；然而SVM训练，适用少样本训练，对于训练样本严格，只有当物体全包含进去了才能标注为物体类别，然后训练样本。
>
> SVM训练阶段要把整个物体包含进去才算正样本，但问题是当我们的检测窗口只有部分物体要怎么定义正负样本，作者测试当重叠度小于0.3的时候，标注为负样本。一旦CNN fc7层特征提取出来后将为每个物体类训练一个SVM分类器，当我们用CNN提取2000个候选框，可以得到2000×4096这样的特征向量矩阵，然后只需要把这样的矩阵与N个SVM矩阵点乘，有N个类别就有N个SVM。
>
> 测试阶段：适用SS在测试图片上提取2000个候选框，将每个候选框归一化倒227×227，然后在CNN中正向传播，将最后一层的特征提取出来，然后对于每个类别使用这个类别训练的SVM分类器对提取的特征向量进行打分，得到每个候选框对于这一类的分数，再使用NMS去除相交多余的框，对这些框进行边缘检测得到bbox。

### Q：对于候选框的处理

> 用SS得到的候选框，大小不一，然而CNN对输入图片是要固定的，需要对候选框作处理。paper中试验了两种不同的处理方法
>
> 各向异性缩放：不管图片长宽比例，不管是否扭曲都缩放到227×227
>
> 各项同性缩放：图片扭曲后估计会对CNN训练精度有影响，结合这个又有两种，先扩充后裁剪，先裁剪后扩充。
>
> 最后，作者发现是采用各向异性缩放，padding=16的精度最高。

### Q： 分类器使用的是二分类？输入是什么？

> 单个SVM实现是二分类，由多个SVM组成，总共20个不同的物体加1种背景，共21个SVM。对21个SVM输出结果进行排序，哪个输出最大，候选区域就属于哪一个。
>
> 输入时特征提取器AlexNet的fc6层的输出，回归器的输入是特征提取器AlexNet的pool5输出结果。之所以这样取输入，是因为，分类器不依赖坐标信息，所以取fc6层的结果没有问题，但会过期依赖坐标信息，要输出坐标的修正量，必须取坐标信息还没有丢失前的层，fc6层已经丢失了坐标信息。



> RCNN每个阶段都是独立的，不能进行端到端的训练，不属于同一体系，CNN+SVM+脊回归器，甚至在测试的时候，需要把每个阶段的结果先保存在磁盘，再喂入下一阶段。Proposal阶段相当于每个图片都要进行1k-2k次的CNN。