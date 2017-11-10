#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Suggestive Annotation test procedure

@author: Jarvis ZHANG
@date: 2017/11/10
@framework: Tensorflow
@editor: VS Code
"""

import tensorflow as tf
import networks
import allPath
import data_provider

FLAGS = networks.FLAGS


def eval_once(saver, metrics, summary_writer, summary_op):
    """Run Eval once.
    Args:
        saver: Saver.
        summary_writer: Summary writer.
        summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(allPath.SA_LOG_TRAIN_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored from file: %s" % ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = []
        try:
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(FLAGS.size_per_test / FLAGS.batch_size))
            avg_dice = 0
            step = 0
            while step < num_iter and not coord.should_stop():
                dice, rvd = sess.run(metrics)
                avg_dice += dice_coef
                step += 1

            avg_dice /= step
            print('%s: dice avg = %.3f' % (datetime.now(), avg_dice))
            print('%s: rvd avg = %.3f' % (datetime.now(), avg_dice))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='dice_coef', simple_value=avg_dice)
            summary.value.add(tag='rvd', simple_value=rvd)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval validation dataset for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels.
        images, labels = data_provider.input(eval_data=True, batch_size=1)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, _ = networks.inference(images, train=False)

        dice_op = networks.dice_coef(logits, labels)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(networks.FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(allPath.SA_LOG_TEST_DIR, g)

        while True:
            eval_once(saver, dice_op, summary_writer, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    # if tf.gfile.Exists(allPath.SA_LOG_TEST_DIR):
    #     tf.gfile.DeleteRecursively(allPath.SA_LOG_TEST_DIR)
    # tf.gfile.MakeDirs(allPath.SA_LOG_TEST_DIR)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
