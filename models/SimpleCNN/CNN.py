#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of a simple convolutional neuron networks

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

# Super parameters
LEARNING_RATE = 1e-4
TRAINING_STEPS = 20000
BATCH_SIZE = 50
VALIDATION_STEP = 100
DROPOUT = 0.5

""" Define the calculation graph """
def new_weight(shape):
    ''' Create new weight variable with stdandard deviation of 0.1
    ### Params:
        * shape - list: shape of the weights
    ### Return:
        * weight - variable: new weights
    '''
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def new_bias(shape):
    ''' Create new bias variable.   
    Biases are initialized by 0.1 to avoid dead neurons.
    ### Params:
        * shape - list: shape of the biases
    ### Return:
        * bias - variable: new biases
    '''
    return tf.Variable(tf.constant(0.1, shape=shape))


# Define the placeholders
x = tf.placeholder(tf.float32, [None, 784])
y_label = tf.placeholder(tf.float32, [None, 10])
dropout_rate = tf.placeholder(tf.float32)
x_img = tf.reshape(x, [-1, 28, 28, 1]) # -1 means uncertain number of samples

# Networks: conv, pooling, output layers
conv1 = tf.nn.relu(tf.nn.conv2d(
    x_img,
    new_weight([5, 5, 1, 32]),
    strides=[1, 1, 1, 1],
    padding='SAME'
) + new_bias([32]))

pool1 = tf.nn.max_pool(
    conv1,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME'
)

conv2 = tf.nn.relu(tf.nn.conv2d(
    pool1,
    new_weight([5, 5, 32, 64]),
    strides=[1, 1, 1, 1],
    padding='SAME'
) + new_bias([64]))

pool2 = tf.nn.max_pool(
    conv2,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME'
)

flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
fc1 = tf.nn.relu(tf.matmul(
    flat,
    new_weight([7 * 7 * 64, 1024])
) + new_bias([1024]))

drop = tf.nn.dropout(fc1, dropout_rate)

y = tf.nn.softmax(tf.matmul(
    drop,
    new_weight([1024, 10])
) + new_bias([10]))


# Loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(
    y_label * tf.log(y), reduction_indices=[1]
))

# Optimizer
step_train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# Accurancy
bool_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1))
accurancy = tf.reduce_mean(tf.cast(bool_prediction, tf.float32))

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
        dropout_rate: DROPOUT
    })

    # Validation
    if i % VALIDATION_STEP == 0:
        accu = accurancy.eval({
            x: mnist.validation.images, 
            y_label: mnist.validation.labels,
            dropout_rate: 1.0
        })
        print("Step: {:0>5}  Validation accurancy: {:.4f}".format(i, accu))
        f.write("Step: {:0>5}  Validation accurancy: {:.4f}\n".format(i, accu))

# Testing
accu = accurancy.eval({
        x: mnist.test.images, 
        y_label: mnist.test.labels,
        dropout_rate: 1.0
    })
print("Final test accurancy: {:.4f}".format(accu))
f.write("Final test accurancy: {:.4f}\n".format(accu))


f.close()
