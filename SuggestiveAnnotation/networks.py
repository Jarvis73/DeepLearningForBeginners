#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Suggestive annotation implementation

@author: Jarvis ZHANG
@date: 2017/10/1
@framework: Tensorflow
@editor: VS Code
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import allPath
import helpers
import data_provider
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# model parameters
tf.app.flags.DEFINE_integer('num_layers', 6, """ Number of layers in model. """)
tf.app.flags.DEFINE_integer('feature_root', 32, """ Feature root. """)
tf.app.flags.DEFINE_integer('batch_size', 1, """ Number of images to process in a batch. """)
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of output classes. """)
tf.app.flags.DEFINE_float('weight_decay', 1e-4, """ Weight decay for L2 loss. """)
tf.app.flags.DEFINE_float('moving_average_decay', 0.9995, """ The decay to use for the moving average. """)
tf.app.flags.DEFINE_float('init_lr', 0.05, """ Initial learning rate. """)
tf.app.flags.DEFINE_float('decay_rate', 0.1, """ Final learning rate. """)
tf.app.flags.DEFINE_integer('epochs', 150, """ Training epochs. """)
tf.app.flags.DEFINE_integer('log_frequency', 50, """ Logging frequency, steps """)
tf.app.flags.DEFINE_string('log_dir', allPath.SA_LOG_TRAIN_DIR, """ Logging directory for training. """)
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('test_frequency', 600, """ Test frequency, steps """)
tf.app.flags.DEFINE_integer('size_per_test', 100, """ Number of examples per test """)
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 10, """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', True, """ Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('dump_number', 10, """ Number of the images to save when testing""")

IMAGE_HEIGHT = data_provider.IMAGE_HEIGHT
IMAGE_WIDTH = data_provider.IMAGE_WIDTH
IMAGE_THICK = data_provider.IMAGE_THICK
IMAGE_DEPTH = data_provider.IMAGE_DEPTH

# File path or constant parameters
NUM_EPOCHES_PER_DECAY = 100


def inference(train=True):
    """ Build the FCN model
    """

    images = tf.placeholder(dtype=tf.float32, shape=[
        FLAGS.batch_size, IMAGE_THICK, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH
    ])
    labels = tf.placeholder(dtype=tf.int32, shape=[
        FLAGS.batch_size, IMAGE_THICK, IMAGE_HEIGHT, IMAGE_WIDTH, 1
    ])

    layer_in = images
    feature_out = FLAGS.feature_root / 2
    upconv_multi = 1

    # keep the output of each stage
    bridges = []

    for layer in range(FLAGS.num_layers):
        # Pooling
        if layer % 2 == 0:
            with tf.name_scope('max_pool{}'.format(layer)) as name:
                pool = tf.layers.max_pooling3d(layer_in, 2, 2, padding='same', name=name)
                helpers.activation_summary(pool)

            layer_in = pool
        
        # Convolution
        feature_out *= 2 if layer % 2 == 0 and layer < 8 else 1
        if layer < 2:
            with tf.variable_scope('conv{}_ord'.format(layer)) as scope:    # Convolution ordinary
                conv = tf.layers.conv3d(layer_in, feature_out, (3, 3, 3), padding='same', name=scope.name + '/conv')
                layer_out = helpers.bn_relu(conv, train, name=scope.name + '/relu')
        else: # bottleneck
            with tf.variable_scope('conv{}_bot'.format(layer)) as scope:    # Convolution bottleneck
                br = helpers.bn_relu(layer_in, train, name=scope.name + '/conv1_NC')
                output_1 = tf.layers.conv3d(br, feature_out, (1, 1, 1), padding='same', name=scope.name + '/conv1_NC')
                output_1 = helpers.bn_relu(output_1, train, name=scope.name + '/conv3_NC')
                output_1 = tf.layers.conv3d(output_1, feature_out, (3, 3, 3), padding='same', name=scope.name + '/conv3_NC')
                output_1 = helpers.bn_relu(output_1, train, name=scope.name + '/conv1_4NC')
                output_1 = tf.layers.conv3d(output_1, feature_out * 4, (1, 1, 1), padding='same', name=scope.name + '/conv1_4NC')

            with tf.variable_scope('conv{}_sct'.format(layer)) as scope:    # Convolution skip connect
                output_2 = tf.layers.conv3d(br, feature_out * 4, (1, 1, 1), padding='same', name=scope.name + '/conv1_4NC_sc')
                
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
                upconv = helpers.bn_relu(upconv, train, name=scope.name)

                bridges.append(upconv)

            helpers.activation_summary(upconv)

    # # output
    # descriptor = tf.reduce_mean(layer_in, axis=[1, 2, 3])
    
    outputs = tf.concat(bridges, axis=-1)
    with tf.variable_scope('output_x3') as scope:
        output_x3 = tf.layers.conv3d(outputs, FLAGS.num_classes, (3, 3, 3), padding='same', name=scope.name)
        output_x3 = helpers.bn_relu(output_x3, train, name=scope.name)
    with tf.variable_scope('output_x1') as scope:
        logits = tf.layers.conv3d(output_x3, FLAGS.num_classes, (1, 1, 1), padding='same', name=scope.name)

    return images, labels, logits
    # return logits


def dice_coef(logits, labels, axis=[1, 2, 3, 4], loss_type='jaccard', epsilon=1e-5, threshold=None):
    """ Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    ### Params:
        * logits: Tensor, inference output with softmax, shape [batch_size, t, h, w, 2]
        * labels: Tensor, actual labels, same size
        * loss_type : string, `jaccard` or `sorensen`, default is `jaccard`.
        * axis : list of integer, All dimensions are reduced, default `[1, 2, 3, 4]`.
        * epsilon : float
            This small value will be added to the numerator and denominator.
            If both output and target are empty, it makes sure dice is 1.
            If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``,
            then if smooth is very small, dice close to 0 (even the image values lower than the threshold),
            so in this case, higher smooth can have a higher dice.
    """
    _, sparse_logits = tf.split(tf.nn.softmax(logits), 2, -1)
    if threshold is not None:
        threshold_tensor = tf.ones_like(sparse_logits) * threshold
        sparse_logits = tf.cast(tf.greater_equal(sparse_logits, threshold_tensor), tf.float32)

    sparse_labels = tf.cast(labels, tf.float32)

    intersection = tf.reduce_sum(sparse_labels * sparse_logits, axis=axis)

    if loss_type == 'jaccard':
        left = tf.reduce_sum(sparse_logits * sparse_logits, axis=axis)
        right = tf.reduce_sum(sparse_labels * sparse_labels, axis=axis)
    elif loss_type == 'sorensen':
        left = tf.reduce_sum(sparse_logits, axis=axis)
        right = tf.reduce_sum(sparse_labels, axis=axis)
    else:
        raise Exception("Unknown loss_type")

    dice = (2.0 * intersection + epsilon) / (left + right + epsilon)

    return tf.reduce_mean(dice)


def loss_cross_entropy(logits, labels):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, thick, num_classes].
      labels: Labels tensor, int32 - [batch_size, width, height, thick, num_classes].
          The ground truth of your data.
    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(tf.squeeze(labels, squeeze_dims=[4]), logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return cross_entropy_mean


def loss(logits, labels):
    """ Combine the softmax cross entropy with L2 loss
    """
    # Calculate L2-loss of all trainable variables
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    
    cross_entropy = loss_cross_entropy(logits, labels)
    tf.add_to_collection('cross_entropy', cross_entropy)

    total_loss = tf.add(cross_entropy, l2_loss * FLAGS.weight_decay, name='total_loss')
    return total_loss


def show_pred(logits):
    # Index 1 of fuse layers correspond to foreground, so discard index 0.
    _, sparse_logits = tf.split(tf.cast(tf.nn.softmax(logits), tf.float32), 2, 4)

    tf.summary.image('prediction', data_provider.flat_cube_tensor(sparse_logits, 8, 8))

    return sparse_logits


def get_pred(logits):
    _, sparse_logits = tf.split(tf.cast(tf.nn.softmax(logits), tf.float32), 2, 4)
    return sparse_logits


def predict(logits, images_ph, labels_ph, model_path, test_images):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        restore(sess, model_path)

        shape = test_images.shape
        shape[-1] = 1
        label_dummy = np.empty(shape)
        prediction = sess.run(get_pred(logits), feed_dict={images_ph: test_images, labels_ph: label_dummy})

    return prediction


def add_loss_summaries(total_loss):
    """Add summaries for losses.
    Generates moving average for all losses and associated summaries for visualizing the performance of the network.

    ### Params:
        * total_loss: Total loss from loss().
    ### Returns:
        * loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    cross_entropy = tf.get_collection('cross_entropy')
    loss_averages_op = loss_averages.apply(cross_entropy + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in cross_entropy + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + '_raw', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """ Train the FCN model
    Create an optimizer and apply to all trainable variables. Add moving average for all trainable variables.

    ### Params:
        * total_loss: Total loss from loss().
        * global_step: Integer Variable counting the number of training steps processed.
    ### Returns:
        * train_op: op for training.
    """
    num_batches_per_epoch = data_provider.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(NUM_EPOCHES_PER_DECAY * num_batches_per_epoch)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.init_lr, 
                                    global_step, 
                                    decay_steps, 
                                    FLAGS.decay_rate, 
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        grads = opt.compute_gradients(total_loss, var_list=tf.trainable_variables())

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    # variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Add control dependencies of batch normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    joint_ops = tf.get_collection('MY_DEPEND')
    with tf.control_dependencies(update_ops + joint_ops + [apply_gradient_op]):
        train_op = tf.no_op(name='train')

    return train_op


def normal_input(istrain):
    images_tensor, labels_tensor = data_provider.distorted_input(eval_data=istrain, batch_size=FLAGS.batch_size)
    sess = tf.Session()
    images, labels = sess.run([images_tensor, labels_tensor])
    return images, labels


def distorted_input(istrain):
    images_tensor, labels_tensor = data_provider.distorted_input(eval_data=istrain, batch_size=FLAGS.batch_size)
    sess = tf.Session()
    images, labels = sess.run([images_tensor, labels_tensor])
    return images, labels


def save(sess, saver, model_path, global_step):
    save_path = saver.save(sess, model_path, global_step)
    return save_path