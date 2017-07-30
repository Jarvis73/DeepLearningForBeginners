#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of softmax regression with input layer and 
output layer. The formula is :
        y = softmax(Wx ＋　b)

@author: Jarvis ZHANG
@date: 2017/7/28
@framework: Tensorflow
@editor: VS Code
"""

import os
import sys
import numpy as np
import tensorflow as tf
from loadData import mnist, N_SAMPLES, show_data_shape

show_data_shape()

# Super parameters
LEARNING_RATE = 0.5
TRAINING_STEPS = 20001
BATCH_SIZE = 100
TEST_STEP = 1000

""" Define the calculation graph """ 
# Define placeholder
x = tf.placeholder(tf.float32, [None, 784])
y_label = tf.placeholder(tf.float32, [None, 10])

# Networks: weights and biases
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Networks: output layer
y = tf.nn.softmax(tf.add(tf.matmul(x, W), b))

# Loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y),
                            reduction_indices=[1]))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
step_train = optimizer.minimize(cross_entropy)

# Open a graph session
f = open(os.path.join(sys.path[0], "output"), "w")
with tf.Session() as sess:
    # Initialize global variables
    tf.global_variables_initializer().run()

    # Training
    for i in range(TRAINING_STEPS):
        img_batch, lab_batch = mnist.train.next_batch(BATCH_SIZE)
        step_train.run({x: img_batch, y_label: lab_batch})

        if i % TEST_STEP == 0:
            bool_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1))
            accurancy = tf.reduce_mean(tf.cast(bool_prediction, tf.float32))
            accu = accurancy.eval({x: mnist.test.images, y_label: mnist.test.labels})
            print("Step: {:0>5}  Test accurancy: {:.4f}".format(i, accu))
            f.write("Step: {:0>5}  Test accurancy: {:.4f}\n".format(i, accu))

f.close()

