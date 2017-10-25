#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper functions for suggestive annotation

@author: Jarvis ZHANG
@date: 2017/10/1
@framework: Tensorflow
@editor: VS Code
"""

import re
import math
import ntpath
import allPath
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from bs4 import BeautifulSoup

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


def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)


def check_space_range():
    mhdpaths = glob(ntpath.join(allPath.LUNA16_RAW_SRC_DIR, "*.mhd"))
    pattern = re.compile(r"ElementSpacing = ([\.\d]*) (?:[\.\d]*) ([\.\d]*)")
    
    spacingXY = []
    spacingZ  = []
    for mhdpath in mhdpaths:
        with open(mhdpath, "r") as f:
            context = f.read()
        result = pattern.search(context)
        spacingXY.append(float(result.group(1)))
        spacingZ.append(float(result.group(2)))

    print("XYMax: ", max(spacingXY), "index: ", np.where(np.array(spacingXY) == max(spacingXY))[0].tolist())
    print("XYMin: ", min(spacingXY), "index: ", np.where(np.array(spacingXY) == min(spacingXY))[0].tolist())
    print("XYMean: ", sum(spacingXY) / len(spacingXY))
    print("ZMax: ", max(spacingZ), "index: ", np.where(np.array(spacingZ) == max(spacingZ))[0].tolist())
    print("ZMin: ", min(spacingZ), "index: ", np.where(np.array(spacingZ) == min(spacingZ))[0].tolist())
    print("ZMean: ", sum(spacingZ) / len(spacingZ))
    print("Length: ", len(spacingXY))
    
    plt.subplot(121)
    plt.hist(spacingXY, 50, normed=True)
    plt.title("Voxel spacing(x, y) distribution")
    plt.subplot(122)
    plt.hist(spacingZ, 50, normed=True)
    plt.title("Voxel spacing(z) distribution")
    plt.show()


def find_xml(pid):
    all_dirs = [d for d in glob(ntpath.join(allPath.LIDC_XML_DIR, "*")) if ntpath.isdir(d)]

    for anno_dir in all_dirs:
        print(anno_dir)
        xml_paths = glob(ntpath.join(anno_dir, "*.xml"))
        for xml_path in xml_paths:
            with open(xml_path, "r") as xml_file:
                markup = xml_file.read()
            xml = BeautifulSoup(markup, features="xml")
            if xml.LidcReadMessage is None:
                continue
            patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text

            if pid in patient_id:
                print(xml_path)
                return


if __name__ == '__main__':
    if False:
        check_space_range()

    if True:
        pid = '268992195564407418480563388746'
        find_xml(pid)

