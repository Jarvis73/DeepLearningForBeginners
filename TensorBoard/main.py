#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tensorboard test

@author: Jarvis ZHANG
@date: 2017/8/1
@framework: Tensorflow
@editor: VS Code
"""
import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorboard import *

# hyper-parameters
MAX_STEPS = 1000
BATCH_SIZE = 100
LEARNING_RATE = 0.001
DROPOUT = 0.9

log_dir = os.path.join(sys.path[0], "log_dir")


# Define input placeholder and reshape
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name='image-input')
    y_label = tf.placeholder(tf.float32, [None, 10], name='label-input')
with tf.name_scope('input_reshape'):
    reshaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', reshaped_input, 10)

# Define hidden layer
hidden1 = nn_layer(x, 784, 500, 'layer1')
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    dropped = tf.nn.dropout(hidden1, keep_prob)

# Define output layer
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

# Choice loss function
with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_label)
    cross_entropy = tf.reduce_mean(diff, name='total')
    tf.summary.scalar('cross_entropy_trend', cross_entropy)

# Choice optimizer
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# Calculate accuracy
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1), name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.summary.scalar('accuracy_trend', accuracy)

# Open a tensorflow interactive session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Merge all the summary
merged = tf.summary.merge_all()

# Define file writer to record the training and testing information
train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(log_dir, 'test'))

# Define a saver
saver = tf.train.Saver()

# Training
feed_list = [x, y_label, keep_prob]
for i in range(MAX_STEPS):
    if i % 10 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False, feed_list=feed_list))
        test_writer.add_summary(summary, i)
        print('step %d: accuracy: %.4f' % (i, acc))
    else:
        if i % 100 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True, feed_list=feed_list),
                                   options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step %03d' % i)
            train_writer.add_summary(summary, i)
            saver.save(sess, os.path.join(log_dir, "model.ckpt"), i)
            print('Adding run metadata for ', i)
        else:
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True, feed_list=feed_list))
            train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()
