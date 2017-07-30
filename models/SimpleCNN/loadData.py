#! /usr/bin/env python
#-*- coding: utf-8 -*-

"""
Load MNIST data sets.

@author: Jarvis ZHANG
@date: 2017/7/30
@framework: Tensorflow
"""
import os
# 禁止tensorflow显示　需要编译tensorflow库
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from prettytable import PrettyTable as ptable
from tensorflow.examples.tutorials.mnist import input_data

# Load mnist data sets
mnist = input_data.read_data_sets('/home/jarvis/DataSet/MNIST_data', one_hot=True)

N_SAMPLES = int(mnist.train.num_examples)

def show_data_shape():
    table = ptable(["", "images", "labels"])
    table.add_row(["training", mnist.train.images.shape, mnist.train.labels.shape])
    table.add_row(["test", mnist.test.images.shape, mnist.test.labels.shape])
    table.add_row(["validation", mnist.validation.images.shape, mnist.validation.labels.shape])
    print(table)


