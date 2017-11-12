#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data provider

@author: Jarvis ZHANG
@date: 2017/11/5
@framework: Tensorflow
@editor: VS Code
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import matplotlib.pyplot as plt
import allPath
import tensorflow as tf
import helpers
import numpy as np

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_THICK = 64
IMAGE_DEEPTH = 1
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1200
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 150
NUM_PREPROCESS_THREADS = 16


def read_from_queue(src_path, rows, cols, size, channels=1, dtype=tf.uint8):
    """ Read 8x8 block image and return the 3D cube
    ### Params:
        * src_path: a Tensor of type string, file path
        * channels: integer, image channels
        * rows: integer, row number of the image block
        * cols: integer, col number of the image block
        * size: integer, size of the image block (square block)
        * dtype: tf type

    ### Returns:
        Tensor, image cube with shape [row*col, size, size]
    """
    image8x8 = tf.image.decode_png(src_path, channels=channels, dtype=dtype)

    cube = None
    for row in range(rows):
        for col in range(cols):
            src_y = row * size
            src_x = col * size

            slice = image8x8[src_y:src_y + size, src_x:src_x + size]
            slice = tf.expand_dims(slice, axis=0)
            if src_x == 0 and src_y == 0:
                cube = slice
            else:
                cube = tf.concat([cube, slice], axis=0)
           
    return cube


def read_data_from_disk(all_files_queue):
    """Reads and parses examples from data files.
    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    ### Args:
        * `all_files_queue`: A queue of strings with the filenames to read from.
    ### Returns:
        * An object representing a single example, with the following fields:
            label: a [height, width, 2] uint8 Tensor with contours tensor in depth 0 and
                segments tensor in depth 1.
            uint8image: a [height, width, depth] uint8 Tensor with the image data
    """
    class Luna16Record(object):
        pass
    
    result = Luna16Record()

    # Read a record, getting filenames from the filename_queue.
    text_reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_content = text_reader.read(all_files_queue)
    o_path, m_path = tf.decode_csv(csv_content, record_defaults=[[""]] * 2)

    result.uint8image = read_from_queue(tf.read_file(o_path), 8, 8, 64)
    result.label = read_from_queue(tf.read_file(m_path), 8, 8, 64)

    return result


def get_read_input(eval_data):
    """ Fetch input data row by row from CSV files.
    ### Args:
        * `eval_data`: integer, representing whether to read from train, validation or test directories.
    ### Returns:
        * `label`: Label of type tf.float32.
        * `image`: Image of type tf.float32, reshaped to correct dimensions.
    """
    # Create queues that produce the filenames and labels to read.
    if eval_data == 0:
        pref = 'train'
    elif eval_data == 1:
        pref = 'validation'
    elif eval_data == 2:
        pref = 'test'
    else:
        raise ValueError("Wrong parameter: eval_data")

    all_files_queue = tf.train.string_input_producer([os.path.join(allPath.SA_ROOT_DIR, pref + '.csv')])

    read_input = read_data_from_disk(all_files_queue)
    image = tf.cast(read_input.uint8image, tf.float32)
    label = tf.cast(read_input.label, tf.int32)

    return label, image


def flat_cube_tensor(cube_tensor, rows, cols):
    """
    ### Params:
        * cub_tensor: Tensor [batch_size, thick, height, width, depth]
        * rows: integer
        * cols: integer
    ### Return:
        Tensor [batch_size, rows*height, cols*width, depth]
    """
    assert rows * cols == cube_tensor.get_shape()[1]
    shape = cube_tensor.get_shape().as_list()
    batch_size = shape[0]
    img_h = shape[2]
    img_w = shape[3]
    img_d = shape[4]
    res_img = tf.Variable(tf.constant(1.0, shape=(batch_size, img_h*rows, img_w*cols, img_d)), trainable=False)
    for row in range(rows):
        for col in range(cols):
            tar_y = row * img_h
            tar_x = col * img_w
            joint = tf.assign(res_img[:, tar_y:tar_y+img_h, tar_x:tar_x+img_w, :], cube_tensor[:, row * cols + col, ...])
            tf.add_to_collection('MY_DEPEND', joint)
    
    return res_img


def _generate_image_and_label_batch(ori_image, std_image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    ### Args:
        * `ori_image`: 3-D Tensor of [height, width, depth] of type.float32, original image
        * `std_image`: 3-D Tensor of [height, width, depth] of type.float32, standard image
        * `label`: 3-D Tensor of [height, width, 1] of type.int32.
        * `min_queue_examples`: int32, minimum number of samples to retain
            in the queue that provides of batches of examples.
        * `batch_size`: Number of images per batch.
        * `shuffle`: boolean indicating whether to use a shuffling queue.
    ### Returns:
        * `images`: Images. 5D tensor of [batch_size, thick, height, width, depth] size.
        * `labels`: Labels. 5D tensor of [batch_size, thick, height, width, depth] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = NUM_PREPROCESS_THREADS
    if shuffle:
        ori_images, std_images, labels = tf.train.shuffle_batch(
            [ori_image, std_image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        ori_images, std_images, labels = tf.train.batch(
            [ori_image, std_image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', flat_cube_tensor(ori_images, 8, 8))
    tf.summary.image('labels', flat_cube_tensor(tf.expand_dims(labels[..., 1], -1), 8, 8))

    return std_images, labels


def input(eval_data, batch_size):
    """Construct input using the Reader ops.
    ### Params:
        * eval_data: bool, indicating if one should use the train or eval data set.
        * batch_size: Number of images per batch.
    ### Returns:
        * images: Images. 4D tensor of [batch_size, IMAGE_THICK, IMAGE_SIZE, IMAGE_SIZE,IMAGE_DEPTH] size.
        * labels: Labels. 4D tensor of [batch_size, IMAGE_THICK, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH] size.
    """
    if not eval_data:
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    
    label, image = get_read_input(eval_data)

    # # Divide the center 32x32x32
    # ox, oy, oz = IMAGE_THICK // 2, IMAGE_HEIGHT // 2, IMAGE_WIDTH // 2
    # image = image[ox:ox+IMAGE_THICK, oy:oy+IMAGE_HEIGHT, oz:oz+IMAGE_WIDTH]
    # label = label[ox:ox+IMAGE_THICK, oy:oy+IMAGE_HEIGHT, oz:oz+IMAGE_WIDTH]

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = helpers.per_image_standardization(image)

    # Set max intensity to 1
    int_label = tf.cast(tf.divide(label, 255), tf.int32)
    int_label = tf.squeeze(int_label, axis=-1)
    int_label = tf.one_hot(int_label, depth=2)

    # Set the shapes of tensors.
    # tf.train.batch or tf.train.shuffle_batch requires all shapes of the tensor must be fully defined
    # image: [thick ? ? depth] --> [thick, height, width, depth]
    image.set_shape([IMAGE_THICK, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEEPTH])
    float_image.set_shape([IMAGE_THICK, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEEPTH])
    int_label.set_shape([IMAGE_THICK, IMAGE_HEIGHT, IMAGE_WIDTH, 2])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    shuffle = False if eval_data else True
    return _generate_image_and_label_batch(image, float_image, int_label,
                                           min_queue_examples, batch_size,
                                           shuffle=shuffle)


def distorted_input(eval_data, batch_size):
    pass


if __name__ == '__main__':
    images, labels = input(False, 1)
    image = flat_cube_tensor(images, 8, 8)
    label = flat_cube_tensor(labels, 8, 8)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = []
        flag = True

        try:
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            depend = tf.get_collection('MY_DEPEND')
            with tf.control_dependencies(depend):
                joint_op = tf.no_op()

            while flag and not coord.should_stop():
                # Run training steps or whatever
                image_out, label_out, _ = sess.run([image, label, joint_op])
                flag = False
                
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
    
    print(np.max(image_out), np.min(image_out))
    print(np.max(label_out), np.min(label_out))

    plt.subplot(121)
    plt.imshow(image_out[0, ..., 0], cmap='gray')
    plt.subplot(122)
    plt.imshow(label_out[0, ..., 1], cmap='gray')
    plt.show()
