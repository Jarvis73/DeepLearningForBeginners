#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
UNet implementation

@author: Jarvis ZHANG
@date: 2017/9/5
@framework: Tensorflow
@editor: VS Code
"""

from datetime import datetime
import math
import time
import platform

import tensorflow as tf

import DCAN

FLAGS = tf.app.flags.FLAGS

if "Windows" in platform.system():
    tf.app.flags.DEFINE_string('eval_dir', 'C:\\Logging\\DCAN\\test\\',
                           """Directory where to write event logs.""")
    tf.app.flags.DEFINE_string('checkpoint_dir', 'C:\\Logging\\DCAN\\train\\',
                           """Directory where to read model checkpoints.""")
elif "Linux" in platform.system():
    tf.app.flags.DEFINE_string('eval_dir', '/home/jarvis/Logging/DCAN/test/',
                           """Directory where to write event logs.""")
    tf.app.flags.DEFINE_string('checkpoint_dir', '/home/jarvis/Logging/DCAN/train/',
                           """Directory where to read model checkpoints.""")

tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 60 * 2,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 80,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")


def eval_once(saver, dice_op, summary_writer, summary_op):
    """Run Eval once.
    Args:
        saver: Saver.
        summary_writer: Summary writer.
        summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
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
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            avg_c_dice = 0
            avg_s_dice = 0
            step = 0
            while step < num_iter and not coord.should_stop():
                c_dice, s_dice = sess.run(dice_op)
                avg_c_dice += c_dice
                avg_s_dice += s_dice
                step += 1

            avg_c_dice /= step
            avg_s_dice /= step
            print('%s: c_dice avg = %.3f' % (datetime.now(), avg_c_dice))
            print('%s: s_dice avg = %.3f' % (datetime.now(), avg_s_dice))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='dice_c', simple_value=avg_c_dice)
            summary.value.add(tag='dice_s', simple_value=avg_s_dice)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval WQU for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for WQU.
        eval_data = FLAGS.eval_data == 'test'
        images, labels = DCAN.inputs(eval_data=eval_data)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        c_fuse, s_fuse = DCAN.inference(images, train=False)

        dice_op = DCAN.dice_op(c_fuse, s_fuse, labels)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(DCAN.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, dice_op, summary_writer, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
