#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
A modified AlexNet on cifar-10 data sets

@author: Jarvis ZHANG
@date: 2017/7/30
@framework: Tensorflow
@editor: VS Code
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf


# Super parameters
TRAINING_STEP = 3000
BATCH_SIZE = 128


