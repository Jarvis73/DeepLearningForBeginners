# Tensorflow 实现的多层感知机

## Introduction
之前用 `Tensorflow` 实现了一个完整的无隐含层的　`Softmax Regression`，并在 `MNIST` 数据集上实现了大约92%的准确率。　

现在要给神经网络加一个隐含层，并使用
* `Dropout` 减轻过拟合
* `Adagrad` 自适应学习率
* `ReLU`　解决梯度弥散问题

没有隐含层的 `Softmax Regression` 只能直接从图像的像素点推断哪个是数字，而没有特征抽象的过程。  
多层神经网络依靠隐含层，可以组合出高阶的特征，比如横线，竖线，圆圈等等，之后可以将这些高阶特征再组合成数字，就能实现精准的匹配和分类。

## Networks
* 输入 784
* 隐层 300
* 输出 10

## Conclusion
得到了98%的准确率
