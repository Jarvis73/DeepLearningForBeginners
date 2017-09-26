#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
UNet implementation

@author: Jarvis ZHANG
@date: 2017/8/15
@framework: Tensorflow
@editor: VS Code
@based on: https://github.com/jakeret/tf_unet
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import sys
import shutil
import numpy as np
from collections import OrderedDict
from time import time, strftime, localtime
import tensorflow as tf

import utils as util2
import outer_utils as util
from layers import (weight_variable, weight_variable_devonc, bias_variable, 
                    conv2d, deconv2d, max_pool, crop_and_concat, pixel_wise_softmax_2,
                    cross_entropy)

# Define some file path
log_dir = "log_dir"
prediction_dir = "prediction"
training_log = strftime("%Y%m%d%H%M%S", localtime(time())) + '.log'
log_path = os.path.join(sys.path[0], log_dir)
log_file = os.path.join(sys.path[0], training_log)
prediction_path = os.path.join(sys.path[0], prediction_dir)

# Define a infomation logger
logger = util2.create_logger(log_file)


def create_conv_net(x, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3, pool_size=2, summaries=True):
    """ Creates a new convolutional unet for the given parametrization.
    ### Params:
        * x - Tensor: shape [batch_size, height, width, channels]
        * keep_prob - float: dropout probability
        * channels - integer: number of channels in the input image
        * n_class - integer: number of output labels
        * layers - integer: number of layers in the net
        * features_root - integer: number of features in the first layer
        * filter_size - integer: size of the convolution filter
        * pool_size - integer: size of the max pooling operation
        * summaries - bool: Flag if summaries should be created
    """
    
    logger.info("Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(layers=layers,
                                                                                                        features=features_root,
                                                                                                        filter_size=filter_size,
                                                                                                        pool_size=pool_size))
    
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    x_image = tf.reshape(x, tf.stack([-1,nx,ny,channels]))
    in_node = x_image
    batch_size = tf.shape(x_image)[0]
    
    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_convs = OrderedDict()
    up_convs = OrderedDict()
    
    # Record the size difference 
    in_size = 1000
    size = in_size

    # Encode
    for layer in range(0, layers):
        features = 2**layer*features_root
        stddev = np.sqrt(2 / (filter_size**2 * features))
        if layer == 0:
            w1 = weight_variable([filter_size, filter_size, channels, features], stddev)
        else:
            w1 = weight_variable([filter_size, filter_size, features//2, features], stddev)
            
        w2 = weight_variable([filter_size, filter_size, features, features], stddev)
        b1 = bias_variable([features])
        b2 = bias_variable([features])
        
        conv1 = conv2d(in_node, w1, keep_prob)
        tmp_h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(tmp_h_conv, w2, keep_prob)
        dw_convs[layer] = tf.nn.relu(conv2 + b2)
        
        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))
        
        size -= 4
        if layer < layers-1:
            pools[layer] = max_pool(dw_convs[layer], pool_size)
            in_node = pools[layer]
            size /= 2
        
    in_node = dw_convs[layers-1]
    
    # Decode
    for layer in range(layers-2, -1, -1):
        features = 2**(layer+1)*features_root
        stddev = np.sqrt(2 / (filter_size**2 * features))
        
        wd = weight_variable_devonc([pool_size, pool_size, features//2, features], stddev)
        bd = bias_variable([features//2])
        h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
        h_deconv_concat = crop_and_concat(dw_convs[layer], h_deconv)
        deconv[layer] = h_deconv_concat
        
        w1 = weight_variable([filter_size, filter_size, features, features//2], stddev)
        w2 = weight_variable([filter_size, filter_size, features//2, features//2], stddev)
        b1 = bias_variable([features//2])
        b2 = bias_variable([features//2])
        
        conv1 = conv2d(h_deconv_concat, w1, keep_prob)
        h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(h_conv, w2, keep_prob)
        in_node = tf.nn.relu(conv2 + b2)
        up_convs[layer] = in_node

        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))
        
        size *= 2
        size -= 4

    # Output Map
    weight = weight_variable([1, 1, features_root, n_class], stddev)
    bias = bias_variable([n_class])
    conv = conv2d(in_node, weight, tf.constant(1.0))
    output_map = tf.nn.relu(conv + bias)
    up_convs["out"] = output_map
    
    # Summary the results of convolution and pooling
    if summaries:
        with tf.name_scope("summary_conv"):
            for i, (c1, c2) in enumerate(convs):
                tf.summary.image('layer_%02d_01'%i, get_image_summary(c1))
                tf.summary.image('layer_%02d_02'%i, get_image_summary(c2))
        
        with tf.name_scope("summary_max_pooling"):
            for k in pools.keys():
                tf.summary.image('pool_%02d'%k, get_image_summary(pools[k]))
        
        with tf.name_scope("summary_deconv"):
            for k in deconv.keys():
                tf.summary.image('deconv_concat_%02d'%k, get_image_summary(deconv[k]))

        with tf.name_scope("down_convolution"):
            for k in dw_convs.keys():
                tf.summary.histogram("layer_%02d"%k + '/activations', dw_convs[k])

        with tf.name_scope("up_convolution"):
            for k in up_convs.keys():
                tf.summary.histogram("layer_%s"%k + '/activations', up_convs[k])
    
    # Record all the variables which can be used in L2 regularization
    variables = []
    for w1,w2 in weights:
        variables.append(w1)
        variables.append(w2)
        
    for b1,b2 in biases:
        variables.append(b1)
        variables.append(b2)

    
    return output_map, variables, int(in_size - size)


class Unet(object):
    """ A unet implementation
    ### Params:
        * channels - integer: (optional) number of channels in the input image
        * n_class - integer: (optional) number of output labels
        * cost - string: (optional) name of the cost function. Default is 'cross_entropy'
        * cost_kwargs - dict: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
        * summaries - bool: keyword arguments. Create summaries or not
    """
    
    def __init__(self, channels=3, n_class=2, cost="cross_entropy", cost_kwargs={}, **kwargs):
        # Clears the default graph stack and resets the global default graph
        tf.reset_default_graph()
        
        self.n_class = n_class
        self.summaries = kwargs.get("summaries", True)
        
        # Placeholder for the input image
        self.x = tf.placeholder("float", shape=[None, None, None, channels])
        self.y = tf.placeholder("float", shape=[None, None, None, n_class])
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        
        logits, self.variables, self.offset = create_conv_net(self.x, self.keep_prob, channels, n_class, **kwargs)
        
        self.cost = self._get_cost(logits, cost, cost_kwargs)
        
        self.gradients_node = tf.gradients(self.cost, self.variables)
         
        self.cross_entropy = tf.reduce_mean(cross_entropy(tf.reshape(self.y, [-1, n_class]),
                                                          tf.reshape(pixel_wise_softmax_2(logits), [-1, n_class])))
        
        self.predicter = pixel_wise_softmax_2(logits)
        self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        
    def _get_cost(self, logits, cost_name, cost_kwargs):
        """ Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        ### Params:
            * logits - Tensor: shape [batch_size, height, width, n_class]
            * cost_name - string: name of the cost function
            * class_weights: weights for the different classes in case of multi-class imbalance
            * regularizer: power of the L2 regularizers added to the loss function
        """
        
        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(self.y, [-1, self.n_class])
        if cost_name == "cross_entropy":
            class_weights = cost_kwargs.pop("class_weights", None)
            
            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
        
                weight_map = tf.multiply(flat_labels, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)
        
                loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                   labels=flat_labels)
                weighted_loss = tf.multiply(loss_map, weight_map)
        
                loss = tf.reduce_mean(weighted_loss)
                
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, 
                                                                              labels=flat_labels))
        elif cost_name == "dice_coefficient":
            eps = 1e-5
            prediction = pixel_wise_softmax_2(logits)
            intersection = tf.reduce_sum(prediction * self.y)
            union =  eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
            loss = -(2 * intersection/ (union))
            
        else:
            raise ValueError("Unknown cost function: "%cost_name)

        regularizer = cost_kwargs.pop("regularizer", None)
        if regularizer is not None:
            regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
            loss += (regularizer * regularizers)
            
        return loss

    def predict(self, model_path, x_test):
        """ Uses the model to create a prediction for the given data
        ### Params:
            * model_path - string: path to the model checkpoint to restore
            * x_test - Tensor: Data to predict on. Shape [n, nx, ny, channels]
        ### return:
            prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2) 
        """
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
        
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            
            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})
            
        return prediction
    
    def save(self, sess, model_path):
        """ Saves the current session to a checkpoint
        ### Params:
            * sess - Session: current session
            * model_path - string: path to file system location
        ### Return:
            save path
        """
        
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path
    
    def restore(self, sess, model_path):
        """ Restores a session from a checkpoint
        ### Params:
            * sess - Session: current session instance
            * model_path - string: path to file system checkpoint location
        """
        
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logger.info("Model restored from file: %s" % model_path)

class Trainer(object):
    """ Trains a unet instance
    ### Params:
        * net - Unet: the unet instance to train
        * batch_size - integer: size of training batch
        * test_batch_size - integer: size of test batch
        * optimizer - string: (optional) name of the optimizer to use (momentum or adam)
        * kwargs - dict: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    """
    
    def __init__(self, net, batch_size=1, test_batch_size=4, optimizer="momentum", opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        self.test_batch_size = test_batch_size
        
    def _get_optimizer(self, training_iters, global_step):
        """ Create the optimizer 
        ### Params:
            * training_iters - integer: learning rate decay steps
            * global_step - integer: global training step
        ### Return:
            Training optimizer
        """
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)
            
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate, 
                                                        global_step=global_step, 
                                                        decay_steps=training_iters,  
                                                        decay_rate=decay_rate, 
                                                        staircase=True)
            
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.cost, 
                                                                                global_step=global_step)
        elif self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            self.learning_rate_node = tf.Variable(learning_rate)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, 
                                               **self.opt_kwargs).minimize(self.net.cost,
                                                                     global_step=global_step)
        
        return optimizer
        
    def _initialize(self, training_iters, log_path, restore):
        """ Initialize the session and summary
        ### Params:
            * training_iters - integer: learning rate decay steps
            * log_path - string: logging directory
            * restore - bool: restore or not
        ### Return:
            global variables initializer
        """
        global_step = tf.Variable(0)
        
        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]))
        self.optimizer = self._get_optimizer(training_iters, global_step)
        init = tf.global_variables_initializer()
        
        with tf.name_scope("summary_gradients"):
            if self.net.summaries:
                tf.summary.histogram('norm_grads', self.norm_gradients_node)
        with tf.name_scope("All_scalars"):
            tf.summary.scalar('cross_entropy', self.net.cross_entropy)
            tf.summary.scalar('learning_rate', self.learning_rate_node)
            tf.summary.scalar('accuracy', self.net.accuracy)
            tf.summary.scalar('loss', self.net.cost)
        self.summary_op = tf.summary.merge_all()        

        if not restore:
            logger.info("Removing '{:}'".format(prediction_path))
            shutil.rmtree(prediction_path, ignore_errors=True)
            logger.info("Removing '{:}'".format(log_path))
            shutil.rmtree(log_path, ignore_errors=True)
        
        if not os.path.exists(prediction_path):
            logger.info("Allocating '{:}'".format(prediction_path))
            os.makedirs(prediction_path)
        
        if not os.path.exists(log_path):
            logger.info("Allocating '{:}'".format(log_path))
            os.makedirs(log_path)
        
        return init

    def train(self, data_provider, training_iters=10, epochs=100, dropout=0.75, display_step=1, restore=False, write_graph=False):
        """ Lauches the training process
        ### Params:
            * data_provider - DataProvider: callable returning training and verification data
            * training_iters - integer: number of training mini batch iteration
            * epochs - integer: number of epochs
            * dropout - float: dropout probability
            * display_step - integer: number of steps till outputting stats
            * restore - bool: Flag if previous model should be restored 
            * write_graph - bool: Flag if the computation graph should be written as protobuf file to the output path
        """
        save_path = os.path.join(log_path, "model.cpkt")
        if epochs == 0:
            return save_path
        
        init = self._initialize(training_iters, log_path, restore)
        
        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, log_path, "graph.pb", False)
            
            sess.run(init)
            
            if restore:
                ckpt = tf.train.get_checkpoint_state(log_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)
            
            test_x, test_y = data_provider(self.test_batch_size)
            pred_shape = self.store_prediction(sess, test_x, test_y, "_init")
            
            summary_writer = tf.summary.FileWriter(log_path, graph=sess.graph)
            logger.info("Start optimization")
            
            avg_gradients = None
            for epoch in range(epochs):
                total_loss = 0
                for step in range((epoch*training_iters), ((epoch+1)*training_iters)):
                    batch_x, batch_y = data_provider(self.batch_size)
                    
                    # Run optimization op (backprop)
                    _, loss, lr, gradients = sess.run((self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node), 
                                                      feed_dict={self.net.x: batch_x,
                                                                 self.net.y: util.crop_to_shape(batch_y, pred_shape),
                                                                 self.net.keep_prob: dropout})

                    # Record gradients
                    if avg_gradients is None:
                        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
                    for i in range(len(gradients)):
                        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step+1)))) + (gradients[i] / (step+1))
                        
                    norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                    self.norm_gradients_node.assign(norm_gradients).eval()
                    
                    # Display steps
                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step, batch_x, util.crop_to_shape(batch_y, pred_shape))
                        
                    total_loss += loss

                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                self.store_prediction(sess, test_x, test_y, "epoch_%s"%epoch)
                    
                save_path = self.net.save(sess, save_path)
            logger.info("Optimization Finished!")
            
            return save_path
        
    def store_prediction(self, sess, batch_x, batch_y, name):
        prediction = sess.run(self.net.predicter, feed_dict={self.net.x: batch_x, 
                                                             self.net.y: batch_y, 
                                                             self.net.keep_prob: 1.})
        pred_shape = prediction.shape
        
        loss = sess.run(self.net.cost, feed_dict={self.net.x: batch_x, 
                                                       self.net.y: util.crop_to_shape(batch_y, pred_shape), 
                                                       self.net.keep_prob: 1.})
        
        logger.info("Verification error= {:.1f}%, loss= {:.4f}".format(error_rate(prediction,
                                                                          util.crop_to_shape(batch_y,
                                                                                             prediction.shape)),
                                                                          loss))
              
        img = util.combine_img_prediction(batch_x, batch_y, prediction)
        util.save_image(img, "%s/%s.jpg"%(prediction_path, name))
        
        return pred_shape
    
    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logger.info("Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))
    
    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss, acc, predictions = sess.run([self.summary_op, 
                                                            self.net.cost, 
                                                            self.net.accuracy, 
                                                            self.net.predicter], 
                                                           feed_dict={self.net.x: batch_x,
                                                                      self.net.y: batch_y,
                                                                      self.net.keep_prob: 1.})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logger.info("Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(step,
                                                                                                                   loss,
                                                                                                                   acc,
                                                                                                                   error_rate(predictions, batch_y)))


def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """
    
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
        (predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))


def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """
    
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255
    
    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V
