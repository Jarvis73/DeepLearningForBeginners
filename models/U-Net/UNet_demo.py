#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
UNet demo

@author: Jarvis ZHANG
@date: 2017/8/2
@framework: Tensorflow
@editor: VS Code
"""

import os
import sys
from UNet import *
import tensorflow as tf
from time import time, strftime, localtime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("bs", 10, 
                            """ Batch size. """)
tf.app.flags.DEFINE_integer("training_steps", 100,
                            """ Training steps. """)
tf.app.flags.DEFINE_integer("img_width", 572,
                            """ Image width. """)
tf.app.flags.DEFINE_integer("img_height", 572,
                            """ Image height. """)
tf.app.flags.DEFINE_integer("img_depth", 1,
                            """ Image depth. """)
tf.app.flags.DEFINE_integer("test_samples", 30,
                            """ Number of the test samples. """)
tf.app.flags.DEFINE_float("learning_rate", 0.001,
                            """ Learning rate. """)

train_images, train_labels = next_distorted_patch(FLAGS.bs)
test_images = all_test_images()

x = tf.placeholder(
    tf.float32, 
    shape=[None, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth]
)
y_ = tf.placeholder(
    tf.float32,
    shape=[None, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth]
)

conv11 = convolution(x, "conv11", [3, 3, 1, 64], sd=0.1, padding='VALID')
conv12 = convolution(conv11, "conv12", [3, 3, 64, 64], sd=0.1, padding='VALID')
pool1 = max_pool(conv12, "pool1", [1, 2, 2, 1], [1, 2, 2, 1])
conv21 = convolution(pool1, "conv21", [3, 3, 64, 128], sd=0.1, padding='VALID')
conv22 = convolution(conv21, "conv22", [3, 3, 128, 128], sd=0.1, padding='VALID')
pool2 = max_pool(conv22, "pool2", [1, 2, 2, 1], [1, 2, 2, 1])
conv31 = convolution(pool2, "conv31", [3, 3, 128, 256], sd=0.1, padding='VALID')
conv32 = convolution(conv31, "conv32", [3, 3, 256, 256], sd=0.1, padding='VALID')
pool3 = max_pool(conv32, "conv32", [1, 2, 2, 1], [1, 2, 2, 1])
conv41 = convolution(pool3, "conv41", [3, 3, 256, 512], sd=0.1, padding='VALID')
conv42 = convolution(conv41, "conv42", [3, 3, 512, 512], sd=0.1, padding='VALID')
pool4 = max_pool(conv42, "pool4", [1, 2, 2, 1], [1, 2, 2, 1])
conv51 = convolution(pool4, "conv51", [3, 3, 512, 1024], sd=0.1, padding='VALID')
conv52 = convolution(conv51, "conv52", [3, 3, 1024, 1024], sd=0.1, padding='VALID')

shape42 = conv42.get_shape()
shape32 = conv32.get_shape()
shape22 = conv22.get_shape()
shape12 = conv12.get_shape()

upconv4 = up_conv(conv52, "upconv4", [3, 3, 512, 1024], [FLAGS.bs, shape42[1], shape42[2], 512])
encoder4 = copy_and_crop(conv42, "bridge4", shape42[1], shape42[2])
concat4 = tf.concat([upconv4, encoder4], axis=3)
conv43 = convolution(concat4, "conv43", [3, 3, 1024, 512], sd=0.1, padding='VALID')
conv44 = convolution(conv43, "conv44", [3, 3, 512, 512], sd=0.1, padding='VALID')

upconv3 = up_conv(conv44, "upconv3", [3, 3, 256, 512], [FLAGS.bs, shape32[1], shape32[2], 256])
encoder3 = copy_and_crop(conv32, "bridge3", shape32[1], shape32[2])
concat3 = tf.concat([upconv3, encoder3], axis=3)
conv33 = convolution(concat3, "conv33", [3, 3, 512, 256], sd=0.1, padding='VALID')
conv34 = convolution(conv33, "conv34", [3, 3, 256, 256], sd=0.1, padding='VALID')

upconv2 = up_conv(conv34, "upconv2", [3, 3, 128, 256], [FLAGS.bs, shape22[1], shape22[2], 128])
encoder2 = copy_and_crop(conv22, "bridge2", shape22[1], shape22[2])
concat2 = tf.concat([upconv2, encoder2], axis=3)
conv23 = convolution(concat2, "conv23", [3, 3, 256, 128], sd=0.1, padding='VALID')
conv24 = convolution(conv23, "conv24", [3, 3, 128, 128], sd=0.1, padding='VALID')

upconv1 = up_conv(conv24, "upconv1", [3, 3, 64, 128], [FLAGS.bs, shape12[1], shape12[2], 64])
encoder1 = copy_and_crop(conv12, "bridge1", shape12[1], shape12[2])
concat1 = tf.concat([upconv1, encoder1], axis=3)
conv13 = convolution(concat1, "conv13", [3, 3, 128, 64], sd=0.1, padding='VALID')
conv14 = convolution(conv13, "conv14", [3, 3, 64, 64], sd=0.1, padding='VALID')
conv15 = convolution(conv14, "conv15", [1, 1, 64, 2], sd=0.1)





