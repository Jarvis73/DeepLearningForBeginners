#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
AlexNet implementation

@author: Jarvis ZHANG
@date: 2017/8/1
@framework: Tensorflow
@editor: VS Code
"""
import os
import sys
import time
import numpy as np
import tensorflow as tf
from datetime import datetime

NUM_BATCHES = 100

def print_activations(t, f=None):
    ''' Show the structure of one layer
    ### Params:
        * t - Tensor: activations of one layer
    ### Return:
        * None
    '''
    print(t.op.name, " ", t.get_shape().as_list())
    if f is not None:
        f.write(t.op.name, " ", t.get_shape().as_list())
    return

def convolution(input, name_scope, shape, 
                strides=[1, 1, 1, 1], 
                dtype=tf.float32,
                sd=1.0, 
                const=0.0, 
                padding='SAME'):
    ''' 2D convolution
    ### Params:
        * input - Tensor: input of the convolution layer
        * name_scope - string: name space
        * shape - list: shape of the kernels
        * strides - list: strides of the kernels
        * sd - float: standard deviation of the weights
        * const - float: value of the biases
        * padding - string: the type of padding algorithm to use. "SAME" or "VALID"
    ### Return:
        * conv - Tensor: result of the convolution
        * parameter - list: trainable parameters (weights and biases)
    '''
    parameters = []
    with tf.name_scope(name_scope) as scope:
        kernel = tf.Variable(tf.truncated_normal(shape=shape, dtype=dtype, stddev=sd), name='weights')
        biases = tf.Variable(tf.constant(const, dtype=dtype, shape=[shape[-1]]), name='biases')
        conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(
            input,
            kernel,
            strides=strides,
            padding=padding
        ), biases), name=scope)
        parameters += [kernel, biases]
    return conv, parameters

def normalization(input, name, 
                  depth_radius=5, 
                  bias=1.0, 
                  alpha=None, 
                  beta=None):
    ''' Define the lrn layer
    ### Params:
        * input - Tensor: input of this layer
        * name - string: name of the lrn layer
        * depth_radius - int: half-width of the 1-D normalization window.
        * bias - float: an offset (usually positive to avoid dividing by 0).
        * alpha - float: a scale factor, usually positive.
        * beta - float: an exponent.
    ### Return:
        * norm - Tensor: result of the lrn
    '''
    norm = tf.nn.lrn(
        input, 
        depth_radius=depth_radius, 
        bias=bias, 
        alpha=alpha, 
        beta=beta,
        name=name
    )
    return norm

def max_pool(input, name, ksize, 
             strides=[1, 1, 1, 1], 
             padding='SAME'):
    ''' Define the max pooling layer
    ### Params:
        * input - Tensor: input of this layer
        * name - string: name of the max pooling layer
        * ksize - list: the size of the window for each dimension of the input tensor
        * strides - list: the stride of the sliding window for each dimension of input
        * padding - string: the type of padding algorithm to use. "SAME" or "VALID"
    ### Return:
        * pool - Tensor: result of the max pooling
    '''
    pool = tf.nn.max_pool(
        input, 
        ksize=ksize, 
        strides=strides, 
        padding=padding,
        name=name
    )
    return pool

def fullyConnected(input, shape, 
                   sd=1.0, 
                   wl=0.0, 
                   const=0.0, 
                   activate=True):
    ''' Define the full connected layer
    ### Params:
        * input - Tensor: input of this layer
        * shape - list: the shape of the kernel
        * stddev - float: stdandard deviation of the weights
        * wl - float: coefficient of the L2 loss item
        * activate - bool: active the output or not
    ### Return:
        * fc - Tensor: result of the fully connected networks
    '''
    weight = variable_with_weight_loss(shape=shape, stddev=sd, wl=wl)
    if activate:
        fc = tf.nn.relu((tf.matmul(input, weight) + 
                    tf.Variable(tf.constant(const, shape=[shape[-1]]))))
    else:
        fc = tf.matmul(input, weight) + tf.Variable(tf.constant(const, shape=[shape[-1]]))
    return fc

def time_run(sess, target, info_string, f=None):
    ''' Evaluate execution time of AlexNet
    ### Params:
        * sess - Session: tensorflow session
        * target - : operator or tensor
        * info_string - string:
        * f - file: file to write up the results
    ### Return: 
        * None
    '''
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(NUM_BATCHES + num_steps_burn_in):
        start_time = time.time()
        sess.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if i % 10 == 0:
                print("%s: step %d, duration = %.3f" % 
                        (datetime.now(), i - num_steps_burn_in, duration))
                f.write("%s: step %d, duration = %.3f" % 
                        (datetime.now(), i - num_steps_burn_in, duration) + "\n")
                total_duration += duration
                total_duration_squared += duration**2
    
    mean_time = total_duration / NUM_BATCHES
    var_time = np.sqrt(total_duration_squared / NUM_BATCHES - mean_time**2)
    print("%s: %s across %d steps, %.3f +/- %.3f sec/batch" % 
            (datetime.now(), info_string, NUM_BATCHES, mean_time, var_time))
    f.write("%s: %s across %d steps, %.3f +/- %.3f sec/batch" % 
            (datetime.now(), info_string, NUM_BATCHES, mean_time, var_time) + "\n")

