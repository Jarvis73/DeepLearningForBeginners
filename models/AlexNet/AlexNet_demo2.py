#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
AlexNet evaluation without lrn layers

@author: Jarvis ZHANG
@date: 2017/8/2
@framework: Tensorflow
@editor: VS Code
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from AlexNet import *
from time import time, strftime, localtime

# hyper-parameters
TRAINING_STEP = 0
BATCH_SIZE = 32
LEARNING_RATE = 0
TEST_STEP = 0
N_SAMPLES = 0
N_TEST = 0
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3

# Create fake data
images = tf.Variable(tf.truncated_normal(
    [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3],
    stddev=0.1,
    dtype=tf.float32
))

""" Construct the calculation graph """
parameters = []
# inference
conv1, params = convolution(images, "conv1", shape=[11, 11, 3, 96], strides=[1, 4, 4, 1], sd=0.1)
pool1 = max_pool(conv1, "pool1", ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
print_activations(conv1)
print_activations(pool1)
parameters += params
conv2, params = convolution(pool1, "conv2", shape=[5, 5, 96, 256], sd=0.1)
pool2 = max_pool(conv2, "pool2", ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
print_activations(conv2)
print_activations(pool2)
parameters += params
conv3, params = convolution(pool2, "conv3", shape=[5, 5, 256, 384], sd=0.1)
print_activations(conv3)
parameters += params
conv4, params = convolution(conv3, "conv4", shape=[5, 5, 384, 384], sd=0.1)
print_activations(conv4)
parameters += params
conv5, params = convolution(conv4, "conv5", shape=[5, 5, 384, 256], sd=0.1)
pool5 = max_pool(conv5, "pool5", ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
print_activations(conv5)
print_activations(pool5)
parameters += params


# Open a tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
objective = tf.nn.l2_loss(pool5)
grad = tf.gradients(objective, parameters)

fname = "evaluation_" + strftime("%m%d%H%M%S", localtime(time()))
f = open(os.path.join(sys.path[0], fname), "w")
# Evaluate forward calculation of AlexNet
time_run(sess, pool5, "Forward", f)

# Evaluate backward calculation of AlexNet
time_run(sess, grad, "Forward-backward", f)

f.close()
