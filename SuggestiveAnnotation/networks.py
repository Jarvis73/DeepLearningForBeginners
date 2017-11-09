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
import data_provider

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

IMAGE_HEIGHT = data_provider.IMAGE_HEIGHT
IMAGE_WIDTH = data_provider.IMAGE_WIDTH
IMAGE_THICK = data_provider.IMAGE_THICK


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
            with tf.variable_scope('conv{}_ord'.format(layer)) as scope:    # Convolution ordinary
                conv = tf.layers.conv3d(layer_in, feature_out, (3, 3, 3), padding='same', name=scope.name + 'conv')
                layer_out = helpers.bn_relu(conv, training, name=scope.name)
        else: # bottleneck
            with tf.variable_scope('conv{}_bot'.format(layer)) as scope:    # Convolution bottleneck
                bn_relu = helpers.bn_relu(layer_in, training, name=scope.name)
                output_1 = tf.layers.conv3d(bn_relu, feature_out, (1, 1, 1), padding='same', name=scope.name + 'conv1_NC')
                output_1 = helpers.bn_relu(output_1, training, name=scope.name)
                output_1 = tf.layers.conv3d(output_1, feature_out, (3, 3, 3), padding='same', name=scope.name + 'conv3_NC')
                output_1 = helpers.bn_relu(output_1, training, name=scope.name)
                output_1 = tf.layers.conv3d(output_1, feature_out * 4, (1, 1, 1), padding='same', name=scope.name + 'conv1_4NC')

            with tf.variable_scope('conv{}_sct'.format(laber)) as scope:    # Convolution skip connect
                output_2 = tf.layers.conv3d(bn_relu, feature_out * 4, (1, 1, 1), padding='same', name=scope.name + 'conv1_4NC_sc')
                
            layer_out = tf.add(output_1, output_2)
        
        helpers.activation_summary(layer_out)
        layer_in = layer_out

        # up-convolution
        if layer % 2 == 1:
            upconv_multi *= 2
            with tf.variable_scope('upconv{}'.format(layer // 2)) as scope:
                feature_in = layer_in.get_shape().as_list()[-1]
                w = helpers.get_deconv_filter_normal(shape=[upconv_multi, upconv_multi, upconv_multi, feature_in, feature_in])
                b = tf.get_variable('biases', [feature_in], initializer=tf.zeros_initializer())
                output_shape = [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_THICK, feature_in]
                upconv = helpers.conv3d_transpose(layer_in, w, b, upconv_multi, output_shape, scope)
                upconv = helpers.bn_relu(upconv, training, name=scope.name)

                bridges.append(upconv)

            helpers.activation_summary(upconv)

    # output
    descriptor = tf.reduce_mean(layer_in, axis=[1, 2, 3])
    
    outputs = tf.concat(bridges, axia=-1)
    with tf.variable_scope('output_x3') as scope:
        output_x3 = tf.layers.conv3d(outputs, FLAGS.num_classes, (3, 3, 3), padding='same', name=scope.name)
        output_x3 = helpers.bn_relu(output_x3, training, name=scope.name)
    with tf.variable_scope('output_x1') as scope:
        logits = tf.layers.conv3d(output_x3, FLAGS.num_classes, (1, 1, 1), padding='same', name=scope.name)
    
    return logits, descriptor
    

def dice_coef(logits, labels, epsilon=1e-5):
    """ Compute dice coefficient
    """
    prediction = tf.nn.softmax(logits, dim=3, name='softmax')
    intersection = tf.reduce_sum(prediction * labels)
    union = tf.reduce_sum(prediction) + tf.reduce_sum(labels)
    return tf.reduce_mean((2.0 * intersection + epsilon) / (union + epsilon))


def softmax_cross_entropy(logits, labels):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, thick, num_classes].
      labels: Labels tensor, int32 - [batch_size, width, height, thick, num_classes].
          The ground truth of your data.
    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('softmax_cross_entropy'):
        logits = tf.reshape(logits, (-1, FLAGS.num_classes))
        labels = tf.to_float(tf.reshape(labels, (-1, FLAGS.num_classes)))

        epsilon = tf.constant(value=1e-4)
        softmax = tf.nn.softmax(logits) + epsilon

        cross_entropy = -tf.reduce_sum(labels * tf.log(softmax), reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return cross_entropy_mean


def loss(logits, labels):
    """ Combine the softmax cross entropy with L2 loss
    """
    # Calculate L2-loss of all trainble variables
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

    #dice_loss = 1 - dice_coef(logits, labels)
    #tf.summary.scalar('dice_loss', dice_loss)
    #total_loss = dice_loss + l2_loss * FLAGS.weight_decay
    
    cross_entropy = softmax_cross_entropy(logits, labels)
    tf.summary.scalar('cross_entropy', cross_entropy)
    total_loss = cross_entropy + l2_loss * FLAGS.weight_decay
    
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


