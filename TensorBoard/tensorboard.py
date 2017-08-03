#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tensorboard test

@author: Jarvis ZHANG
@date: 2017/8/1
@framework: Tensorflow
@editor: VS Code
"""

import os
# 禁止tensorflow显示　需要编译tensorflow库
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# hyper-parameters
BATCH_SIZE = 100
DROPOUT = 0.9

# Load mnist data sets
data_dir = "/home/jarvis/DataSet/MNIST_data"
mnist = input_data.read_data_sets(data_dir, one_hot=True)

def variable_summary(var, name):
    ''' Summary the variable
    ### Params:
        * var - Tensor: variable to summary
        * name - string: name scope
    ### Return:
        * None
    '''
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.histogram('histogram', var)

def nn_layer(input, in_dim, out_dim, name, act=tf.nn.relu):
    ''' Define the networks layer
    ### Params:
        * input - Tensor: input of this layer
        * in_dim - integer: input dimention
        * out_dim - intege: output dimention
        * name - string: name scope of this layer
        * act - function: activate function
    ### Return:
        * activations - Tensor: output of this layer
    '''
    with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[out_dim]), name='biases')
        variable_summary(weights, 'w_sum')
        variable_summary(biases, 'b_sum')
        preactivate = tf.matmul(input, weights) + biases
        tf.summary.histogram('pre_act', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('post_act', activations)
        return activations

def feed_dict(train, feed_list):
    ''' Feed dictionary
    ### Params:
        * train - bool: training or testing
        * feed_list - list: feed list
    ### Return:
        * feed - dict: feed dictionary
    '''
    if train:
        images, labels = mnist.train.next_batch(BATCH_SIZE)
        kp = DROPOUT
    else:
        images, labels = mnist.test.next_batch(BATCH_SIZE)
        kp = 1.0
    return {feed_list[0]: images, feed_list[1]: labels, feed_list[2]: kp}




