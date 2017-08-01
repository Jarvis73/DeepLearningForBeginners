#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
A modified AlexNet on cifar-10 data sets

@author: Jarvis ZHANG
@date: 2017/8/1
@framework: Tensorflow
@editor: VS Code
"""

import tensorflow as tf

def print_activations(t, f=None):
    ''' Show the structure of one layer
    ### Params:
        * t - tensor: activations of one layer
    ### Return:
        * None
    '''
    print(t.op.name, " ", t.get_shape().as_list())
    if f is not None:
        f.write(t.op.name, " ", t.get_shape().as_list())
    return

