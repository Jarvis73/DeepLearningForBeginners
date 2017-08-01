# Tensorflow 实现一个简单的类似于AlexNet的神经网络

## Introduction
使用CIFAR-10的数据集，包含60000张32*32的真彩色图像。
* 训练集: 50000张
* 测试集: 10000张
该数据集包含了10种不同的种类，包括 airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck。

已有的成果发布在 [classification datasets results](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html) 上。

**Special**
* 对权重进行L2正则化
* 对图片进行翻转，随机剪切等数据增强，制造更多的样本
* 在每个卷积－最大池化层后使用了LRN层，增强了模型的泛化能力。


## Networks
|Layer名称 |描述              |
|:-------|:---------------|
|conv1    |卷基层和ReLU激活函数|
|pool1    |最大池化|
|norm1    |LRN|
|conv2    |卷基层和ReLU激活函数|
|norm2    |LRN|
|pool2    |最大池化|
|fc1      |全连接层和ReLU激活函数|
|fc2      |全连接层和ReLU激活函数|
|logits   |模型Inference的输出结果|

