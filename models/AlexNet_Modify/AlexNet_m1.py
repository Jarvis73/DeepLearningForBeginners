#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
A modified AlexNet on cifar-10 data sets

@author: Jarvis ZHANG
@date: 2017/7/30
@framework: Tensorflow
@editor: VS Code
"""

import tensorflow as tf


def variable_with_weight_loss(shape, stddev, wl):
    ''' Create weight variables with L2 loss
    ### Params:
        * shape - list: shape of the weight arrays
        * stddev - float: stdandard deviation of the weights
        * wl - float: coefficient of the L2 loss item
    ### Return:
        * weight - tensor: weights tensor as convolutional kernels
    '''
    weight = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(weight), wl, name="weight_loss")
        tf.add_to_collection('losses', weight_loss)
    return weight

def convolution(input, shape, strides, sd=1.0, wl=0.0, const=0.0, padding='SAME'):
    ''' Define the convolutional layer
    ### Params:
        * input - Tensor: input of this layer
        * shape - list: the shape of the kernel
        * strides - list: the stride of the sliding window for each dimension of input
        * stddev - float: stdandard deviation of the weights
        * wl - float: coefficient of the L2 loss item
        * padding - string: the type of padding algorithm to use. "SAME" or "VALID"
    ### Return:
        * conv - Tensor: result of the convolution
    '''
    weight = variable_with_weight_loss(shape=shape, stddev=sd, wl=wl)
    conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(
        input, 
        weight, 
        strides=strides, 
        padding=padding
    ), tf.Variable(tf.constant(const, shape=[shape[-1]]))))
    return conv

def max_pool(input, ksize, strides, padding='SAME'):
    ''' Define the max pooling layer
    ### Params:
        * input - Tensor: input of this layer
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
        padding=padding
    )
    return pool

def normalization(input, depth_radius=5, bias=1.0, alpha=None, beta=None):
    ''' Define the lrn layer
    ### Params:
        * input - Tensor: input of this layer
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
        beta=beta
    )
    return norm

def fullyConnected(input, shape, sd=1.0, wl=0.0, const=0.0, activate=True):
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

def loss(logits, labels):
    ''' Calculate the loss of the model with softmax and cross entropy
    ### Params:
        * logits - Tensor: output of the networks
        * labels - Tensor: anaotation of the images
    ### Return:
        * loss - float: total loss of the model
    '''
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')





