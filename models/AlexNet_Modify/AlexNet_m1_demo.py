#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
A modified AlexNet on cifar-10 data sets

@author: Jarvis ZHANG
@date: 2017/8/1
@framework: Tensorflow
@editor: VS Code
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from loadData import summary_cifar_img, next_distorted_batch, next_original_batch
from AlexNet_m1 import *


# Some hyper-parameters
TRAINING_STEP = 3000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
N_TEST = 10000


# Choice next batch of images and labels
images_train, labels_train = next_distorted_batch(BATCH_SIZE)
images_test, labels_test = next_original_batch(BATCH_SIZE)

# Create placeholder
images = tf.placeholder(tf.float32, [BATCH_SIZE, 24, 24, 3])
labels = tf.placeholder(tf.int32, [BATCH_SIZE])

""" Construct calculation graph """
# inference
conv1 = convolution(images, shape=[5, 5, 3, 64], strides=[1, 1, 1, 1], sd=5e-2, wl=0.0, const=0.0, padding='SAME')
pool1 = max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = normalization(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)
conv2 = convolution(norm1, shape=[5, 5, 64, 64], strides=[1,1,1,1], sd=5e-2, wl=0.0, const=0.0, padding='SAME')
norm2 = normalization(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)
pool2 = max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
reshape = tf.reshape(pool2, [BATCH_SIZE, -1])
dim = reshape.get_shape()[1].value
fc1 = fullyConnected(reshape, [dim, 384], sd=0.04, wl=0.004, const=0.1, activate=True)
fc2 = fullyConnected(fc1, shape=[384, 192], sd=0.04, wl=0.004, const=0.1, activate=True)
logits = fullyConnected(fc2, shape=[192, 10], sd=1.0/192, wl=0.0, const=0.0, activate=False)

# loss
loss = loss(logits, labels)

# Optimizer
step_train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

# output
top_k_op = tf.nn.in_top_k(logits, labels, 1)

# Open a session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Start input enqueue threads
tf.train.start_queue_runners()

# Training
f = open(os.path.join(sys.path[0], "output"), 'w')
for step in range(TRAINING_STEP):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run(
        [step_train, loss],
        feed_dict={images: image_batch, labels: label_batch}
    )
    duration = time.time() - start_time

    # Watching
    if step % 20 == 0:
        examples_per_sec = BATCH_SIZE / duration
        sec_per_batch = float(duration)
        format_str = "step {:0>5}: loss={:.4f} ({:.1f} examples/sec; {:.3f} sec/batch)"
        print(format_str.format(step, loss_value, examples_per_sec, sec_per_batch))
        f.write(format_str.format(step, loss_value, examples_per_sec, sec_per_batch) + "\n")

# Testing
num_iter = int(np.ceil(N_TEST / BATCH_SIZE))
true_count = 0
total_sample_count = num_iter * BATCH_SIZE
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run(
        [top_k_op], 
        feed_dict={images: image_batch, labels: label_batch}
    )
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print("Precision @ 1 = %.3f" % precision)
f.write("Precision @ 1 = %.3f" % precision)

f.close()
sess.close()
