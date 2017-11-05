#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Suggestive annotation implementation

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
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of output classes. """)
tf.app.flags.DEFINE_float('weight_decay', 1e-4, """ Weight decay for L2 loss. """)
tf.app.flags.DEFINE_float('init_lr', 0.01, """ Initial learning rate. """)
tf.app.flags.DEFINE_float('decay_rate', 0.1, """ Final learning rate. """)
tf.app.flags.DEFINE_integer('epoches', 100, """ Training epochs. """)
tf.app.flags.DEFINE_integer('log_frequency', 20, """ Logging frequency. """)
tf.app.flags.DEFINE_string('log_dir', "C:\\Logging\\SA\\train", """ Logging directory for training. """)
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")



# File path or constant parameters
NUM_EPOCHES_PER_DECAY = 30


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
                bn_relu = helpers.bn_relu(layer_in, training, name=scope.name)
                output_1 = tf.layers.conv2d(bn_relu, feature_out, (1, 1), padding='same', name=scope.name + 'conv1x1_NC')
                output_1 = helpers.bn_relu(output_1, training, name=scope.name)
                output_1 = tf.layers.conv2d(output_1, feature_out, (3, 3), padding='same', name=scope.name + 'conv3x3_NC')
                output_1 = helpers.bn_relu(output_1, training, name=scope.name)
                output_1 = tf.layers.conv2d(output_1, feature_out * 4, (1, 1), padding='same', name=scope.name + 'conv1x1_4NC')

            with tf.variable_scope('conv{}_sct'.format(laber)) as scope:
                output_2 = tf.layers.conv2d(bn_relu, feature_out * 4, (1, 1), padding='same', name=scope.name + 'conv1x1_4NC_sc')
                
            layer_out = tf.add(output_1, output_2)
        
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
    
    outputs = tf.concat(bridges, axia=3)
    with tf.variable_scope('output_3x3') as scope:
        output_3x3 = tf.layers.conv2d(outputs, FLAGS.num_classes, (3, 3), padding='same', name=scope.name)
    with tf.variable_scope('output_1x1') as scope:
        logits = tf.layers.conv2d(output_3x3, FLAGS.num_classes, (1, 1), padding='same', activation=tf.nn.softmax, name=scope.name)
    
    return logits, descriptor
    

def dice_coef(logits, labels, epsilon=1e-5):
    """ Compute dice coefficient
    """
    prediction = tf.nn.softmax(logits, dim=3, name='softmax')
    intersection = tf.reduce_sum(prediction * labels)
    union = tf.reduce_sum(prediction) + tf.reduce_sum(labels)
    return tf.reduce_mean((2.0 * intersection + epsilon) / (union + epsilon))


def loss(logits, labels):
    """ Combine the dice coefficient with L2 loss
    """
    dice_loss = 1 - dice_coef(logits, labels)
    tf.summary.scalar('dice_loss', dice_loss)
    # Add L2-loss to the total loss
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

    total_loss = dice_loss + l2_loss * FLAGS.weight_decay
    tf.summary.scalar('total_loss', total_loss)
    return total_loss


def train(loss, global_step):
    """ Train the FCN model

    """
    decay_steps = NUM_EPOCHES_PER_DECAY * NUM_EXAMPLES_PER_EPOCH
    lr = tf.train.exponential_decay(FLAGS.init_lr, 
                                    global_step, 
                                    decay_steps, 
                                    FLAGS.decay_rate, 
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Add histograms for trainable variables
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    optimizer = tf.train.MomentumOptimizer(lr, 0.9)

    # Add control dependencies of batch normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)

    return train_op


