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
import time
import logging
import platform
import tensorflow as tf
from tensorflow import gfile

import DCAN

FLAGS = tf.app.flags.FLAGS

if "Windows" in platform.system():
    tf.app.flags.DEFINE_string('train_dir', 'C:\\Logging\\DCAN\\train\\',
                           """Directory where to write event logs and checkpoint.""")
elif "Linux" in platform.system():
    tf.app.flags.DEFINE_string('train_dir', '/home/jarvis/Logging/DCAN/train/',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 40000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 20,
                            """How often to log results to the console.""")
tf.logging.set_verbosity(tf.logging.DEBUG)

def train():
    """Train WQU for a number of steps."""
    if not gfile.Exists(FLAGS.train_dir):
        gfile.MakeDirs(FLAGS.train_dir)

    with tf.Graph().as_default():
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        global_step_init = -1
        global_step = tf.contrib.framework.get_or_create_global_step()
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            global_step_init = int(ckpt.model_checkpoint_path.split('/')[-1]
                                   .split('-')[-1])

        # Get images and labels for WQU.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up
        # on GPU and resulting in a slow down.
        images, labels = DCAN.inputs(eval_data=False)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        c_fuse, s_fuse = DCAN.inference(images)
        DCAN.get_show_preds(c_fuse, s_fuse)

        # Calculate loss.
        loss = DCAN.loss(c_fuse, s_fuse, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = DCAN.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = global_step_init
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        saver = tf.train.Saver()
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement),
                save_checkpoint_secs=1000) as mon_sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

            if ckpt:
                saver.restore(mon_sess, ckpt.model_checkpoint_path)
                logging.info("Model restored from file: %s" % ckpt.model_checkpoint_path)
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    tf.app.run()

