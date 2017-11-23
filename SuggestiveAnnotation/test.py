#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Suggestion Annotation training procedure

@author: Jarvis ZHANG
@date: 2017/11/22
@framework: Tensorflow
@editor: VS Code
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import allPath
import helpers
import networks
import prepare_data
import data_provider

FLAGS = networks.FLAGS
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def test(dump_pred_image=False, num_of_image=None, threshold=None):
    if not tf.gfile.Exists(allPath.SA_LOG_TEST_DIR2):
        tf.gfile.MakeDirs(allPath.SA_LOG_TEST_DIR2)

    # Create a information logger, write output into console and file
    logger = helpers.create_logger(file=False)

    with tf.Graph().as_default():
        # Define input data stream
        images_tensor_for_test, labels_tensor_for_test = data_provider.input(eval_data=2,
                                                                             batch_size=FLAGS.batch_size)
        # Build a Graph that computes the logits predictions from the inference model.
        images_ph, labels_ph, logits = networks.inference(train=False)
        prediction = networks.get_pred(logits)

        dice = networks.dice_coef(logits, labels_ph, threshold=threshold)

        sess = tf.Session()
        with sess.as_default():
            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = []

            saver = tf.train.Saver()
            best_model_path = os.path.join(allPath.SA_LOG_ALL_BEST_DIR, "best_model.ckpt-137400")
            saver.restore(sess, best_model_path)
            try:
                # qr: tf.train.QueueRunner object
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                dice_list = []
                for i in tqdm(range(data_provider.NUM_EXAMPLES_PER_EPOCH_FOR_TEST)):
                    test_images, test_labels = sess.run([images_tensor_for_test, labels_tensor_for_test])

                    dice_value, pred_images = sess.run([dice, prediction],
                                                       feed_dict={images_ph: test_images, labels_ph: test_labels})
                    dice_list.append(dice_value)

                    if dump_pred_image and \
                            (num_of_image is None or num_of_image is not None and i < num_of_image):
                        prepare_data.save_cube_img(os.path.join(allPath.SA_LOG_TEST_DIR2, "%d_ori.png" % i),
                                                   test_images[0, ..., 0] * 255, 8, 8)
                        prepare_data.save_cube_img(os.path.join(allPath.SA_LOG_TEST_DIR2, "%d_lab.png" % i),
                                                   test_labels[0, ..., 0] * 255, 8, 8)
                        prepare_data.save_cube_img(os.path.join(allPath.SA_LOG_TEST_DIR2, "%d_pre.png" % i),
                                                   pred_images[0, ..., 0] * 255, 8, 8)

                    i += 1

                avg_dice = np.mean(dice_list)
                std_dice = np.std(dice_list)
                logger.info("avg_dice: %.3f pm %.3f" % (avg_dice, std_dice))


            except Exception as e:
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


if __name__ == '__main__':
    # eps = 1e-3
    # for i in np.arange(eps, 1-eps, 0.1):
    #     test(threshold=i)
    test(threshold=0.2, dump_pred_image=False, num_of_image=30)