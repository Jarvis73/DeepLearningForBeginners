#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Load cifar-10 data set

@author: Jarvis ZHANG
@date: 2017/7/30
@framework: Tensorflow
@editor: VS Code
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# hyper-parameters
BATCH_SIZE = 32
NUM_BATCHES = 100

