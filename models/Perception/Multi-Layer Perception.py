#! /usr/bin/env python
#-*- coding: utf-8 -*-

"""
Implementation of Multi-Layer Perception with one hidden layer.

@author: Jarvis ZHANG
@date: 2017/7/30
@framework: Tensorflow
@editor: VS Code
"""

import os
import sys
import numpy as np
import tensorflow as tf
from loadData import mnist, N_SAMPLES, show_data_shape

show_data_shape()

# Some hyper-parameters
LEARNING_RATE = 0.3
TRAINING_STEPS = 3000
BATCH_SIZE = 100
TEST_STEP = 100
DROPOUT = 0.75

""" Define the calculation graph """

# Define the placeholders
x = tf.placeholder(tf.float32, [None, 784])
y_label = tf.placeholder(tf.float32, [None, 10])
dropout = tf.placeholder(tf.float32)

# Networks: weights and biases
W1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1))
b1 = tf.Variable(tf.zeros([300]))
W2 = tf.Variable(tf.zeros([300, 10]))
b2 = tf.Variable(tf.zeros([10]))

# Networks: hidden and output layers
hidden = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
hidden_drop = tf.nn.dropout(hidden, dropout)
y = tf.nn.softmax(tf.add(tf.matmul(hidden_drop, W2), b2))

# Loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(
    y_label * tf.log(y), reduction_indices=[1]
))

# Optimizer
step_train = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(cross_entropy)

# accuracy
bool_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(bool_prediction, tf.float32))

""" Begin to train the networks """

# Open a graph session
sess = tf.InteractiveSession()

# Initialize global variables
tf.global_variables_initializer().run()

# Traning
f = open(os.path.join(sys.path[0], "output"), "w")
for i in range(TRAINING_STEPS):
    img_batch, lab_batch = mnist.train.next_batch(BATCH_SIZE)
    step_train.run({
        x: img_batch, 
        y_label: lab_batch,
        dropout: DROPOUT
    })

    # Testing
    if i % TEST_STEP == 0:
        accu = accuracy.eval({
            x: mnist.test.images, 
            y_label: mnist.test.labels,
            dropout: 1.0
        })
        print("Step: {:0>5}  Test accuracy: {:.4f}".format(i, accu))
        f.write("Step: {:0>5}  Test accuracy: {:.4f}\n".format(i, accu))

f.close()
