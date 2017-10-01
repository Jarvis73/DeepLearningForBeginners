#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
UNet implementation

@author: Jarvis ZHANG
@date: 2017/10/1
@framework: Tensorflow
@editor: VS Code
"""

import math
import tensorflow as tf

def activation_summary(x):
    """ Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    ### Params:
        * x: Tensor
    """
    
    tf.summary.histogram(x.op.name + '/activations', x)
    tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def get_deconv_filter(shape):
    """Return deconvolution weight tensor w/bilinear interpolation.
    Args:
        shape: 4D list of weight tensor shape.
    Returns:
        Tensor containing weight variable.

    Source:
        https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn16_vgg.py#L245
    """
    height = shape[0]
    width = shape[1]
    f = math.ceil(width / 2.0)
    c = (2.0 * f - 1 - f % 2) / (2.0 * f)

    bilinear = np.zeros([shape[0], shape[1]])
    for x in range(width):
        for y in range(height):
            bilinear[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))

    weights = np.zeros(shape)
    for i in range(shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return tf.get_variable(name='up_filter', initializer=init, shape=weights.shape)


def bn_relu(inputs, training, name=None):
    """ Batch normalization & ReLU
    Note: when training, the moving_mean and moving_variance need to be updated. 
          By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be added as a dependency to the train_op. 
          For example:
    
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(loss)
    
    """

    bn = tf.layers.batch_normalization(inputs, training=training, name=name + 'normalization')
    layer_out = tf.nn.relu(bn, name=name + 'relu')
    return layer_out


def conv2d_transpose(inputs, w, b, stride, output_shape, scope):
    """Return deconvolution layer tensor.
    ### Params:
        * inputs: Input tensor layer.
        * w: Weight tensor.
        * b: Bias tensor.
        * stride: Deconvolution constant.
        * output_shape: Deconvolution layer output shape in list format.
        * scope: Enclosing variable scope.
    ### Returns:
        Tensor for deconvolution layer.
    """
    deconv = tf.nn.conv2d_transpose(inputs, w, output_shape, strides=[1, stride, stride, 1],
                                    padding='SAME')
    deconv = tf.nn.bias_add(deconv, bias=b, name=scope.name)
    return deconv




