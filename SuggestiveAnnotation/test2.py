#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Suggestive Annotation test procedure

@author: Jarvis ZHANG
@date: 2017/11/10
@framework: Tensorflow
@editor: VS Code
"""

import os
import cv2
import math
import time
import networks
import allPath
import logging
import prepare_data
import data_provider
import numpy as np
import tensorflow as tf

from datetime import datetime


FLAGS = networks.FLAGS
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def evaluate():
    """Eval validation dataset for a number of steps."""
    with tf.Graph().as_default():
        ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
        global_step_init = -1
        global_step = tf.contrib.framework.get_or_create_global_step()
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            global_step_init = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])

        # Get images and labels.
        ori_images, std_images, labels = data_provider.input(eval_data=True, batch_size=1)

        # Build a Graph that computes the logits predictions from the inference model.
        logits = networks.inference(std_images, train=False)

        dice_op = networks.dice_coef(logits, labels, loss_type='sorensen')

        # train_op = networks.train(loss, global_step)
        test_op = networks.test(global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                # Parameters for Session.run()
                return tf.train.SessionRunArgs(fetches=[ori_images, labels, logits, dice_op])

            def after_run(self, run_context, run_values):
                print(run_values.results[3])
                # Save images

                prepare_data.save_cube_img(os.path.join(allPath.SA_LOG_TEST_IMG_DIR, "%d_ori.png" % self._step),
                                           run_values.results[0][0, ..., 0], 8, 8)
                prepare_data.save_cube_img(os.path.join(allPath.SA_LOG_TEST_IMG_DIR, "%d_lab.png" % self._step),
                                           run_values.results[1][0, ..., 1] * 255, 8, 8)
                prepare_data.save_cube_img(os.path.join(allPath.SA_LOG_TEST_IMG_DIR, "%d_pre.png" % self._step),
                                           run_values.results[2][0, ..., 1] * 255, 8, 8)

        saver = tf.train.Saver()
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=allPath.SA_LOG_TEST_DIR,
                hooks=[tf.train.StopAtStepHook(last_step=100),
                       tf.train.NanTensorHook(dice_op),
                       _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement),
                save_checkpoint_secs=None, save_summaries_steps=1) as mon_sess:

            ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)

            if ckpt:
                saver.restore(mon_sess, ckpt.model_checkpoint_path)
                logging.info("Model restored from file: %s" % ckpt.model_checkpoint_path)
            while not mon_sess.should_stop():
                mon_sess.run(test_op)


def main(argv=None):
    if tf.gfile.Exists(allPath.SA_LOG_TEST_IMG_DIR):
        tf.gfile.DeleteRecursively((allPath.SA_LOG_TEST_IMG_DIR))
    tf.gfile.MakeDirs(allPath.SA_LOG_TEST_IMG_DIR)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
