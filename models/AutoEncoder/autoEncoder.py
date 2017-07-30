#! /usr/bin/env python
#-*- coding: utf-8 -*-

"""
Implementation of Denoising AutoEncoder with one hidden layer.

@author: Jarvis ZHANG
@date: 2017/7/28
@framework: Tensorflow
"""

import numpy as np
import tensorflow as tf

def xavier_init(fan_in, fan_out, constant=1):
    """参数化初始化方法: xavier initialization
    如果深度学习模型的权重初始化得太小, 那信号将在每层间传递时逐渐缩小而难以产生作用;
    如果权重初始化得太大, 那信号将在每层间传递时逐渐放大并导致发散和失效.
    Xaiver初始化器做的就是让权重被初始化得不大不小正合适.
    数学上来说就是让权重为0, 方差为2/(n_in + n_out)
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval = low, 
                            maxval=high, dtype=tf.float32)

class AdditiveGuassianNoiseAutoEncoder(object):
    def __init__(self, nInput, nHidden, activate_function=tf.nn.softplus,
                optimizer=tf.train.AdamOptimizer(), scale=0.1):
        ''' Only one hidden layer
        Params:
            nInput - integer: number of input variables
            nHidden - integer: number of hidden neurons
            activate_function - function: activate function, default softplus
            optimizer - function(): optimizer, default Adam
            scale - float: coefficient of Guassian noise, default 0.1
        '''
        self.nInput = nInput
        self.nOutput = nInput
        self.nHidden = nHidden
        self.activate = activate_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        self.weights = self._initialize_weights()

        # Define the structure of the networks
        self.x = tf.placeholder(tf.float32, [None, self.nInput])
        self.hidden = self.activate(tf.add(tf.matmul(
                                scale * tf.random_normal((nInput,)) + self.x, 
                                self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(
                                self.hidden, self.weights['w2']), 
                                self.weights['b2'])
        
        # Define the cost function: square error
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
                                self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        # Initialize variables and construct session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        ''' Initialize connection weights and biases 
        Params:
            None
        Return:
            all_weights - dict: all the weights and biases
        '''
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.nInput, self.nHidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.nHidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(xavier_init(self.nHidden, self.nOutput))
        all_weights['b2'] = tf.Variable(tf.zeros([self.nOutput], dtype=tf.float32))
        return all_weights

    def step_train(self, train_batch):
        ''' Calculate cost and execute one step train
        Params:
            train_batch - tensor: a train batch
        Return:
            cost - tensor: train cost
        '''
        cost, _ = self.sess.run((self.cost, self.optimizer), 
                            feed_dict={self.x: train_batch, 
                            self.scale: self.training_scale})
        return cost

    def cost_test(self, test_batch):
        ''' Calculate the cost of one test batch on current networks
        Params: 
            test_batch - tensor: a test batch
        Return:
            cost - tensor: test cost
        '''
        return self.sess.run(self.cost,
                            feed_dict={self.x: test_batch, 
                            self.scale: self.training_scale})

    def encoder(self, batch):
        ''' A interface to abtain abstract feature from hidden layer
        Params:
            batch - tensor: input batch
        Return:
            hidden - tensor: output of the hidden layer
        '''
        return self.sess.run(self.hidden,
                            feed_dict={self.x: batch, 
                            self.scale: self.training_scale})

    def decoder(self, hidden=None):
        ''' A interface to abtain recovery data
        Params:
            hidden - tensor: output of the hidden layer
        Return:
            reconstruction - tensor: recovery data
        '''
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction,
                            feed_dict={self.hidden: hidden})

    def reconstruct(self, batch):
        ''' Finish one step of encoding and decoding
        Params: 
            batch - tensor: original data
        Return:
            reconstruction - tensor: recovery data
        '''
        return self.sess.run(self.reconstruction,
                            feed_dict={self.x: batch,
                            self.scale: self.training_scale})

    def getWeights1(self):
        ''' Get the weights of the hidden layer
        Params:
            None
        Return:
            weights - tensor: weights of the hidden layer
        '''
        return self.sess.run(self.weights['w1'])

    def getBiases1(self):
        ''' Get the biases of the hidden layer
        Params:
            None
        Return:
            biases - tensor: biases of the hidden layer
        '''
        return self.sess.run(self.weights['b1'])





