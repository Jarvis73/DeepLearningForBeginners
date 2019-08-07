#!/usr/bin/env python
# coding: utf-8

# ## 使用生成器为 Keras 训练提供数据
# 
# 持续训练模型可能有过拟合的风险, 一种解决方法是提前停止训练, 因此我们可以在训练的过程中加入验证, 当验证精度长时间不再提升的时候就可以停止训练了.

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

# 我们定义一个数据预处理函数. 这里为了简单只做灰度值归一化的操作.

def preprocess(image, label):
    image = (image / 255.).astype(np.float32)
    label = label.astype(np.int32)
    return image, label


# 接下来我们想使用 Keras 的 `fit_generator` 函数来开启训练, 因此要先定义一个数据生成器. 这个数据生成器接受 images 数组和 labels 数组作为输入, 同时可以指定批大小 `batch_size`, 和是否要打乱数据 `shuffle=True/False` 

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

# 这里我们仍然定义包含一个隐藏层的多层感知机(MLP)

model = K.Sequential([
    nn.Flatten(input_shape=(28, 28)),
    nn.Dense(128, activation=tf.nn.relu),
    nn.Dense(10, activation=tf.nn.softmax)
])


# 定义模型优化器, 损失函数和评估指标

model.compile(tf.train.AdamOptimizer(),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# ### 4. 开始训练

# 首先创建用于训练和验证的数据生成器

batch_size = 16
train_gen = data_loader(train_images, train_labels, batch_size=batch_size)
val_gen = data_loader(test_images, test_labels, batch_size=batch_size, shuffle=False)


# 然后我们使用 `fit_generator` 函数, 同时提供训练生成器和验证生成器. Keras 会在每个 epoch 结束的时候评估验证集的数据, 并输出验证集上的准确率.

model.fit_generator(train_gen, steps_per_epoch=train_labels.shape[0] // batch_size, epochs=10,
                    validation_data=val_gen, validation_steps=test_labels.shape[0] // batch_size)


# ### 5. 训练结束后评估
# 
# 这里我们评估也使用生成器

test_loss, test_acc = model.evaluate_generator(val_gen, steps=test_labels.shape[0] // batch_size)
print('Test accuracy:', test_acc)


# 使用数组输入评估, 要记得做同样的数据预处理.

test_images_, test_labels_ = preprocess(test_images, test_labels)
test_loss, test_acc = model.evaluate(test_images_, test_labels_)
print('Test accuracy:', test_acc)


# 结果肯定是一样的.



