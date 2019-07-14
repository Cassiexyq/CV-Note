#### Caffe

1. 数据增强

   crop shift rotate color  投影变换  block（遮挡） noise 模糊不模糊

2. 生成数据集  lmdb  convert_imagset.exe 可以根据情况修改这个代码
3. 生成网络  .net.prototxt
4. get solver  定义超参数 定义网络 定义model存储位置
5. 训练