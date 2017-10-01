#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
UNet implementation

@author: Jarvis ZHANG
@date: 2017/10/1
@framework: Tensorflow
@editor: VS Code
"""

import tensorflow as tf
import helpers

FLAGS = tf.app.flags.FLAGS

# model parameters
tf.app.flags.DEFINE_integer('num_layers', 12, """ Number of layers in model. """)
tf.app.flags.DEFINE_integer('feature_root', 32, """ Feature root. """)
tf.app.flags.DEFINE_integer('batch_size', 1, """ Number of images to process in a batch. """)
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of output classes """)


# File path or constant parameters
IMAGE_HEIGHT = 0
IMAGE_WIDTH = 0



def inference(images, train=True):
    """ Build the FCN model
    """

    # define placeholder
    training = tf.placeholder(dtype=tf.bool, shape=[1])

    layer_in = images
    feature_out = FLAGS.feature_root / 2
    upconv_multi = 1

    # keep the output of each stage
    bridges = []

    for layer in range(FLAGS.num_layers):
        # Pooling
        if layer % 2 == 0:
            with tf.name_scope('max_pool{}'.format(layer)) as name:
                pool = tf.layers.max_pooling2d(layer_in, 2, 2, padding='same', name=name)
                helpers.activation_summary(pool)

            layer_in = pool
        
        # Convolution
        feature_out *= 2 if layer % 2 == 0 and layer < 8 else 1
        if layer < 2:
            with tf.variable_scope('conv{}_ord'.format(layer)) as scope:
                conv = tf.layers.conv2d(layer_in, feature_out, (3, 3), padding='same', name=scope.name + 'conv')
                layer_out = helpers.bn_relu(conv, training, name=scope.name)
        else: # bottleneck
            with tf.variable_scope('conv{}_bot'.format(layer)) as scope:
                conv1x1 = tf.layers.conv2d(layer_in, feature_out, (1, 1), padding='same', name=scope.name + 'conv1x1_NC')
                bn_relu_1 = helpers.bn_relu(conv1x1, training, name=scope.name)
                conv3x3 = tf.layers.conv2d(bn_relu_1, feature_out, (3, 3), padding='same', name=scope.name + 'conv3x3_NC')
                bn_relu_2 = helpers.bn_relu(conv3x3, training, name=scope.name)
                conv1x1 = tf.layers.conv2d(bn_relu_2, feature_out * 4, (1, 1), padding='same', name=scope.name + 'conv1x1_4NC')

            with tf.variable_scope('conv{}_sct'.format(laber)) as scope:
                shortcut = tf.layers.conv2d(layer_in, feature_out * 4, (1, 1), padding='same', name=scope.name + 'conv1x1_4NC_sc')
                
            layer_out = tf.add(conv1x1, shortcut)
            layer_out = helpers.bn_relu(conv1x1, training, name=scope.name)
        
        helpers.activation_summary(layer_out)
        layer_in = layer_out

        # up-convolution
        if layer % 2 == 1:
            upconv_multi *= 2
            with tf.variable_scope('upconv{}'.format(layer // 2)) as scope:
                feature_in = layer_in.get_shape().as_list()[-1]
                w = helpers.get_deconv_filter([upconv_multi, upconv_multi, feature_in, feature_in])
                b = tf.get_variable('biases', [feature_in], initializer=tf.constant_initializer(0.1))
                output_shape = [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, feature_in]
                upconv = helpers.conv2d_transpose(layer_in, w, b, upconv_multi, output_shape, scope)
                upconv = helpers.bn_relu(upconv, training, name=scope.name)

                bridges.append(upconv)

            helpers.activation_summary(upconv)

    # output
    descriptor = tf.reduce_mean(layer_in, axis=[1, 2])
    
    outputs = tf.concat(bridges, 3)
    with tf.variable_scope('output_3x3') as scope:
        output_3x3 = tf.layers.conv2d(outputs, FLAGS.num_classes, (3, 3), padding='same', name=scope.name)
    with tf.variable_scope('output_1x1') as scope:
        logits = tf.layers.conv2d(output_3x3, FLAGS.num_classes, (1, 1), padding='same', activation=tf.nn.softmax, name=scope.name)
    
    return logits, descriptor
    

def dice_coef(logits, labels, epsilon=100):
    logits_bin = tf.cast(tf.equal(logits, 1), tf.float32)
    labels_bin = tf.cast(tf.equal(labels, 1), tf.float32)
    intersection = tf.reduce_sum(tf.multiply(logits_bin, labels_bin))
    union = tf.reduce_sum(logits_bin) + tf.reduce_sum(labels_bin)
    return (2.0 * intersection + 100) / (union + 100)


def loss(logits, labels):
    return 1 - dice_coef(logits, labels)


def train(loss, global_step):
    """ Train the FCN model

    """

    

