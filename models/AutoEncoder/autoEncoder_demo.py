#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test procdure of AutoEncoder on the mnist data sets
Visualization of the mnist data sets and networks output

@author: Jarvis ZHANG
@date: 2017/7/28s
@framework: Tensorflow
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import autoEncoder
from loadData import N_SAMPLES, mnist
from loadData import next_batch, input_normalization, pseudo_batch

# Some super parameters
TRAINING_EPOCHS = 10
BATCH_SIZE = 128
TEST_SIZE = 8
DISPLAY_STEP = 1
LEARNING_RATE = 0.001
TOTAL_BATCH = int(N_SAMPLES / BATCH_SIZE)

# Create instance of AGN AutoEncoder
autoEdr = autoEncoder.AdditiveGuassianNoiseAutoEncoder(
            nInput = 784,
            nHidden = 200,
            activate_function = tf.nn.softplus,
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE),
            scale=2)

# Input normalization
X_train, X_test = input_normalization(mnist.train.images, mnist.test.images)

# Begin training
f = open(os.path.join(sys.path[0], "output"), "w")
_, plots = plt.subplots(TRAINING_EPOCHS // DISPLAY_STEP + 1, 1, figsize=(5, 10))
test_batch = pseudo_batch(X_test, TEST_SIZE, 1234).reshape(TEST_SIZE, 28, 28)
test_batch = np.column_stack(test_batch)
plots[0].axis('off')
plots[0].imshow(test_batch, cmap='gray')

for epoch in range(TRAINING_EPOCHS):
    avg_cost = 0.0
    for i in range(TOTAL_BATCH):
        batch = next_batch(X_train, BATCH_SIZE)
        cost = autoEdr.step_train(batch)
        avg_cost += cost / N_SAMPLES * BATCH_SIZE

    if epoch % DISPLAY_STEP == 0:
        print("Epoch: {:0>4}, cost = {:.9f}".format(epoch + 1, avg_cost))
        f.write("Epoch: {:0>4}, cost = {:.9f}\n".format(epoch + 1, avg_cost))
        # Test
        test_batch = pseudo_batch(X_test, TEST_SIZE, 1234)
        output_rsp = autoEdr.reconstruct(test_batch).reshape(TEST_SIZE, 28, 28)
        output_rsp = np.column_stack(output_rsp)
        plots[epoch // DISPLAY_STEP + 1].axis('off')
        plots[epoch // DISPLAY_STEP + 1].imshow(output_rsp, cmap='gray')

f.close()

# Display input and output
plt.show()
