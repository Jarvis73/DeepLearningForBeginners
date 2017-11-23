#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Suggestion Annotation training procedure

@author: Jarvis ZHANG
@date: 2017/11/5
@framework: Tensorflow
@editor: VS Code
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import time
import tensorflow as tf

import helpers
import networks
import allPath
import data_provider
import prepare_data

FLAGS = networks.FLAGS
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def train():
    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)
    if not tf.gfile.Exists(allPath.SA_LOG_TEST_IMG_DIR):
        tf.gfile.MakeDirs(allPath.SA_LOG_TEST_IMG_DIR)
    if not tf.gfile.Exists(allPath.SA_LOG_BEST_DIR):
        tf.gfile.MakeDirs(allPath.SA_LOG_BEST_DIR)

    # Create a information logger, write output into console and file
    logger = helpers.create_logger()

    with tf.Graph().as_default():
        # Define input data stream
        images_tensor_for_train, labels_tensor_for_train = data_provider.input(eval_data=0,
                                                                               batch_size=FLAGS.batch_size)
        images_tensor_for_eval, labels_tensor_for_eval = data_provider.input(eval_data=1,
                                                                             batch_size=FLAGS.batch_size)

        # Build a Graph that computes the logits predictions from the inference model.
        images_ph, labels_ph, logits = networks.inference(train=True)
        prediction = networks.show_pred(logits)

        # Calculate loss.
        loss = networks.loss(logits, labels_ph)
        dice = networks.dice_coef(logits, labels_ph)

        # Define global step
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = networks.train(loss, global_step)

        # Define model saver
        saver = tf.train.Saver(max_to_keep=5)
        best_saver = tf.train.Saver(max_to_keep=2)

        # Define session
        sess = tf.Session()
        with sess.as_default():
            # Initializer
            sess.run(tf.global_variables_initializer())
    
            # Merge summary
            merged = tf.summary.merge_all()
    
            # Define the summary writer to record the train information
            train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    
            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = []
    
            # Checkpoints restore
            global_step_init = 0
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step_init = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                saver.restore(sess, ckpt.model_checkpoint_path)     # Here saver will log restore information
    
            try:
                # qr: tf.train.QueueRunner object
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
    
                # Training
                best_dice = 0.0
                start_time = time.time()
                max_steps = int(FLAGS.epochs * data_provider.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size)
                for i in range(max_steps - global_step_init):
                    step = i + global_step_init
                    train_images, train_labels = sess.run([images_tensor_for_train, labels_tensor_for_train])
    
                    # One train step
                    _, loss_value = sess.run(fetches=[train_op, loss],
                                             feed_dict={images_ph: train_images, labels_ph: train_labels})
                    if step % FLAGS.log_frequency == 0:
                        current_time = time.time()
                        duration = current_time - start_time
                        examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                        sec_per_batch = float(duration / FLAGS.log_frequency)
                        format_str = 'step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                        logger.info(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

                        summary = sess.run(merged, feed_dict={images_ph: train_images, labels_ph: train_labels})
                        train_writer.add_summary(summary, step)
                        start_time = time.time()
    
                    # Try to validate
                    if step % FLAGS.test_frequency == 0:
                        total_dice = 0
                        for j in range(FLAGS.size_per_test):
                            test_images, test_labels = sess.run([images_tensor_for_eval, labels_tensor_for_eval])
                            dice_value, pred_images = sess.run(fetches=[dice, prediction],
                                                               feed_dict={images_ph: test_images,
                                                                          labels_ph: test_labels})
                            total_dice += dice_value
                            if j < FLAGS.dump_number:
                                prepare_data.save_cube_img(os.path.join(allPath.SA_LOG_TEST_IMG_DIR, "%d_ori.png" % j),
                                                           test_images[0, ..., 0] * 255, 8, 8)
                                prepare_data.save_cube_img(os.path.join(allPath.SA_LOG_TEST_IMG_DIR, "%d_lab.png" % j),
                                                           test_labels[0, ..., 0] * 255, 8, 8)
                                prepare_data.save_cube_img(os.path.join(allPath.SA_LOG_TEST_IMG_DIR, "%d_pre.png" % j),
                                                           pred_images[0, ..., 0] * 255, 8, 8)
                            j += 1

                        avg_dice = total_dice / FLAGS.size_per_test

                        logger.info('dice avg = %.3f' % (avg_dice))

                        # Save model
                        model_name = os.path.join(FLAGS.log_dir, 'model.ckpt')
                        logger.info('Model saved as %s' % (networks.save(sess, saver, model_name, step)))

                        # Save best model
                        if best_dice < avg_dice:
                            best_dice = avg_dice
                            best_model_name = os.path.join(allPath.SA_LOG_BEST_DIR, 'best_model.ckpt')
                            networks.save(sess, best_saver, best_model_name, step)

                        start_time = time.time()
    
                    i += 1
            except Exception as e:
                coord.request_stop(e)
    
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

            train_writer.close()

    
def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
