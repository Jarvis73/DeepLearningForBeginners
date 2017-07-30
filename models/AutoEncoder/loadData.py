#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Load the mnist data sets

@author: Jarvis ZHANG
@date: 2017/7/28
@framework: Tensorflow
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import time
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data

# Load mnist data sets
mnist = input_data.read_data_sets('/home/jarvis/DataSet/MNIST_data', one_hot=True)

N_SAMPLES = int(mnist.train.num_examples)

def input_normalization(X_train, X_test):
    ''' Normalize the input data to standard normal distribution
    Params:
        X_train - tensor: train data
        X_test - tesnor: test data
    Return:
        X_train - tensor: normalized train data
        X_test - tensor: normalized train data
    '''
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def next_batch(data, batch_size):
    ''' Get next batch data
    Params:
        data - tensor: train, validation or test data
        batch_size - integer: batch size
    Return:
        batch - tensor: a batch data
    '''
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

def pseudo_batch(data, batch_size, seed=None):
    ''' Generate a pseudo random number for choising batch data
    Params:
        data - tensor: train, validation or test data
        batch_size - integer: batch size
        seed - float: random seed
    Return:
        batch - tensor: a batch data
    '''
    if seed is None:
        seed = time.time()
    pseudo = np.random.RandomState(seed)
    start_index = pseudo.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

