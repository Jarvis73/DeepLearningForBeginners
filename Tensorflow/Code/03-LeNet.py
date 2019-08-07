#!/usr/bin/env python
# coding: utf-8

# ## 使用 Keras 创建第一个卷积神经网络 LeNet
# 
# 这一部分我们构造一个比多层感知机更为复杂的基于卷积的神经网络--LeNet.
# 

import numpy as np
import tensorflow as tf
from tensorflow import keras as K

print(tf.VERSION, K.__version__)
nn = K.layers


# ### 1. 导入数据

fashion_mnist = K.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)


# ### 2. 数据预处理

# 数据预处理部分与上一节的内容相同

def preprocess(image, label):
    image = (image / 255.).astype(np.float32)
    label = label.astype(np.int32)
    return image, label


def data_loader(images, labels, batch_size=4, shuffle=True):
    assert images.shape[0] == labels.shape[0],         "Shape mismatch: images {} vs labels {}".format(images.shape, labels.shape)
    images, labels = preprocess(images, labels)

    while True:
        all_indices = np.arange(images.shape[0])
        if shuffle:
            np.random.shuffle(all_indices)
        for i in range(0, all_indices.shape[0], batch_size):
            image_batch = images[all_indices[i:i + batch_size]]
            label_batch = labels[all_indices[i:i + batch_size]]
            yield image_batch, label_batch


# ### 3. 定义模型

# 这里我们定义一个卷积神经网络 LeNet, 它的结构为 Conv-Pool-Conv-Pool-Conv-Flatten-FC-FC, 其中
# * Conv: 二维卷积层
# * Pool: 二维池化层
# * Flatten: 把二维图像压平成一维向量
# * FC: 全连接层
# 
# 同时这里我们使用 Keras 的函数式 API, 函数式 API 比序列式 API 的优点是前者可以构造更复杂的网络结构, 如多输入多输出, 层之间的跳跃连接等. 

def get_model():
    inputs = K.Input(shape=(28, 28, 1))                         # 28x28@6
    out = nn.Conv2D(6, 5, activation=tf.nn.relu, padding="same")(inputs)        # 28x28@6
    out = nn.MaxPool2D()(out)                                   # 14x14@6
    out = nn.Conv2D(16, 5, activation=tf.nn.relu)(out)          # 10x10@16
    out = nn.MaxPool2D()(out)                                   # 5x5@16
    out = nn.Conv2D(120, 5, activation=tf.nn.relu)(out)         # 1x1@160
    out = nn.Flatten()(out)                                     # 160
    out = nn.Dense(84, activation=tf.nn.relu)(out)              # 84
    out = nn.Dense(10, activation=tf.nn.softmax)(out)           # 10
    return inputs, out

x, y = get_model()
model = K.Model(inputs=x, outputs=y)


# 定义模型优化器, 损失函数和评估指标. Tensorflow 中的 Keras 模型在编译时既可以直接指定 `tf.train.AdamOptimizer()` 这类 Tensorflow 的优化器(上一节那样), 也可以通过字符串指定优化器, 就像下面这样.

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# ### 4. 开始训练

# 训练部分和上一节一模一样, 我们仅仅修改了模型搭建的部分.
# 首先创建用于训练和验证的数据生成器. 
# 
# **注意:** 这里我们给 `train_images` 和 `test_images` 都增加了一个维度, 其形状变成了 `[60000, 28, 28, 1]` 和 `[10000, 28, 28, 1]`. 这样做的目的是 Keras 中的卷积操作 `Conv2D` 要求数据的输入尺寸为 `[batch_size, height, width, channel]`, 所以我们必须补一维通道. 

batch_size = 16
train_gen = data_loader(train_images[..., None], train_labels, batch_size=batch_size)
val_gen = data_loader(test_images[..., None], test_labels, batch_size=batch_size)


# 然后我们使用 `fit_generator` 函数, 同时提供训练生成器和验证生成器. Keras 会在每个 epoch 结束的时候评估验证集的数据, 并输出验证集上的准确率.

model.fit_generator(train_gen, steps_per_epoch=train_labels.shape[0] // batch_size, epochs=10,
                    validation_data=val_gen, validation_steps=test_labels.shape[0] // batch_size)


# ### 5. 训练结束后评估
# 
# 这里我们评估也使用生成器

test_loss, test_acc = model.evaluate_generator(val_gen, steps=test_labels.shape[0] // batch_size)
print('Test accuracy:', test_acc)


# 可以看到 LeNet 的识别准确率要高于单隐层的神经网络, 有大约0.02的提升.

