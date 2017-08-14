#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
UNet implementation

@author: Jarvis ZHANG
@date: 2017/8/2
@framework: Tensorflow
@editor: VS Code
"""

import os
import sys
import tensorflow as tf


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
        Result of the max pooling
    '''
    pool = tf.nn.max_pool(
        input, 
        ksize=ksize, 
        strides=strides, 
        padding=padding,
        name=name
    )
    return pool

def copy_and_crop(input, name_scope, height, width):
    ''' Copy and crop the image from encoder and send to decoder.
    ### Params:
        * input - Tensor: input of this bridge
        * name_scope - string: name of this bridge
        * shape - list: shape of the feature map after cropping.
                        Not include channels.
    ### Return:
        Cropped feature maps
    '''
    with tf.name_scope(name_scope):
        crop = tf.image.resize_image_with_crop_or_pad(input, height, width)
    return crop

def up_conv(input, name_scope, shape, output_shape,
            strides=[1, 1, 1, 1]
            dtype=tf.float32,
            sd=1.0, 
            padding='SAME'):
    ''' Up sampling with convolution
    ### Params:
        * input - Tensor: input of this layer
        * name_scope - string: name of this layer
        * shape - list: shape of the kernels
        * output_shape - list: shape of the layer's output
        * strides - list: strides of the kernels
        * padding - string: the type of padding algorithm to use. "SAME" or "VALID"
    ### Return:
        Results of the up convolution
    '''
    with tf.name_scope(name_scope) as scope:
        kernel = tf.Variable(tf.truncated_normal(shape=shape, dtype=dtype, stddev=sd), name='weights')
        deconv = tf.nn.relu((tf.nn.conv2d_transpose(
            input,
            kernel,
            output_shape = output_shape
            strides=strides,
            padding=padding
        ), name=scope)
    return deconv
