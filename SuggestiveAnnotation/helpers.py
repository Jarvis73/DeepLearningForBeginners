#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper functions for suggestive annotation

@author: Jarvis ZHANG
@date: 2017/10/1
@framework: Tensorflow
@editor: VS Code
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from glob import glob
from datetime import datetime
from bs4 import BeautifulSoup
from skimage.feature import canny
from time import time, strftime, localtime
from skimage.morphology import dilation, square

import os
import re
import cv2
import math
import logging
import allPath
import numpy as np
import prepare_data
import tensorflow as tf
import matplotlib.pyplot as plt


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
        shape: 5D list of weight tensor shape.
    Returns:
        Tensor containing weight variable.

    Source:
        https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn16_vgg.py#L245
    """
    f = math.ceil(shape[0] / 2.0)
    c = (2.0 * f - 1 - f % 2) / (2.0 * f)

    bilinear = np.zeros([shape[0], shape[1], shape[2]])
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                bilinear[x, y, z] = (1 - abs(x / f - c)) * (1 - abs(y / f - c)) * (1 - abs(z / f - c))

    weights = np.zeros(shape)
    for i in range(shape[-2]):
        weights[..., i, i] = bilinear

    return tf.get_variable(name='up_filter', initializer=tf.constant_initializer(value=weights), 
                            shape=weights.shape)


def get_deconv_filter_normal(shape):
    """Return deconvolution weight tensor w with Gauss distribution.
    Args:
        shape: 5D list of weight tensor shape.
    Returns:
        Tensor containing weight variable.

    """
    return tf.get_variable(name='up_filter', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))


def bn_relu(inputs, training, name=None):
    """ Batch normalization & ReLU
    Note: when training, the moving_mean and moving_variance need to be updated. 
          By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be added as a dependency to the train_op. 
          For example:
    
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(loss)
    """

    bn = tf.layers.batch_normalization(inputs, training=training, name=name + '/normalization')
    layer_out = tf.nn.relu(bn, name=name + '/relu')
    return layer_out


def conv3d_transpose(inputs, w, b, stride, output_shape, scope):
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
    deconv = tf.nn.conv3d_transpose(inputs, w, output_shape, strides=[1, stride, stride, stride, 1],
                                    padding='SAME')
    deconv = tf.nn.bias_add(deconv, bias=b, name=scope.name)
    return deconv


def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, -1, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)


def check_space_range():
    mhdpaths = glob(os.path.join(allPath.LUNA16_RAW_SRC_DIR, "*.mhd"))
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
    all_dirs = [d for d in glob(os.path.join(allPath.LIDC_XML_DIR, "*")) if os.path.isdir(d)]

    for anno_dir in all_dirs:
        print(anno_dir)
        xml_paths = glob(os.path.join(anno_dir, "*.xml"))
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


def per_image_standardization(image):
    """Linearly scales `image` to have zero mean and unit norm.

    This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
    of all values in image, and
    `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.

    `stddev` is the standard deviation of all values in `image`. It is capped
    away from zero to protect against division by 0 when handling uniform images.

    ### Params:
      image: 4-D tensor of shape `[thick, height, width, channels]`.

    ### Returns:
      The standardized image with same shape as `image`.
    """
    num_pixels = tf.reduce_prod(tf.shape(image))
    
    image = tf.cast(image, dtype=tf.float32)
    image_mean = tf.reduce_mean(image)

    variance = (tf.reduce_mean(tf.square(image)) - tf.square(image_mean))
    variance = tf.nn.relu(variance)
    stddev = tf.sqrt(variance)

    min_stddev = tf.rsqrt(tf.cast(num_pixels, tf.float32))
    pixel_value_scale = tf.maximum(stddev, min_stddev)
    pixel_value_offset = image_mean

    image = tf.subtract(image, pixel_value_offset)
    image = tf.div(image, pixel_value_scale)
    return image


def show_contour(image, mask, write_path):
    # Canny edge detection
    contour = canny(image)
    contour = dilation(contour, square(1)).astype(np.uint8) * 255
    
    # Set label red
    image[contour > 0, 2] = 255
    image[contour > 0, 1] = 0
    image[contour > 0, 0] = 0

    cv2.imwrite('color.png', image)


class MyFormatter(logging.Formatter):
    converter = datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s.%03d" % (t, record.msecs)
        return s


def create_logger(log_file=None, file=True, console=True):
    """ Create a logger to write info to console and file.
    ### Params:
        * log_file - string: path to the logging file
    ### Return:
        A logger object of class getLogger()
    """
    if file:
        if log_file is None:
            log_name = strftime('%Y%m%d%H%M%S', localtime(time())) + '.log'
            log_file = os.path.join(allPath.SA_LOGGING_DIR, log_name)

        if os.path.exists(log_file):
            os.remove(log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = MyFormatter("%(asctime)s: %(levelname).1s %(message)s")

    if file:
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        # Register handler
        logger.addHandler(file_handler)

    if console:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


if __name__ == '__main__':
    if False:
        check_space_range()

    if False:
        pid = '168833925301530155818375859047'
        find_xml(pid)

    if True:
        logger = create_logger('./logging.log')
        logger.info("abcdefg")
        logger.warning("12345")

