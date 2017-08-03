#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
U-Net implementation

@author: Jarvis ZHANG
@date: 2017/8/2
@framework: Tensorflow
@editor: VS Code
"""

import os
import sys
import numpy as np
from libtiff import TIFF
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = 512
IMG_SLICES = 30

data_dir = "/home/jarvis/DataSet/ISBI Challenge"
train_images_path = os.path.join(data_dir, "train-volume.tif")
train_labels_path = os.path.join(data_dir, "train-labels.tif")
test_images_path = os.path.join(data_dir, "test-volume.tif")

def _read_data(path):
    container = np.empty((IMG_SLICES, IMG_SIZE, IMG_SIZE))
    # load data
    tif = TIFF.open(path)
    for i, image in enumerate(tif.iter_images()):
        container[i] = image
    tif.close()
    return container

def _transform_data():
    train_images = read_data(train_images_path)
    train_labels = read_data(train_labels_path)
    test_images = read_data(test_images_path)
    writer = tf.python_io.TFRecordWriter("")

def _distorted_images():
    return


def distorted_data():
    train_images, train_labels, test_images = input_data()
