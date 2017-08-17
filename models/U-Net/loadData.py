#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Load tif data

@author: Jarvis ZHANG
@date: 2017/8/2
@framework: Tensorflow
@editor: VS Code
"""

import os
import sys
import errno
import platform
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = 572
IMG_DEPTH = 1
IMG_SLICES = 30
BATCH_SIZE = 10

# Global string defination
train_records = "train_record.tfrecords"
test_records = "test_record.tfrecords"
all_img_lab = "all_img_lab.npz"
linux_data_dir = "/home/jarvis/DataSet/ISBI Challenge"
win_data_dir = "E:\\DataSet\\ISBI Challenge"
# Ensure the operating system
if "Windows" in platform.system():
    data_dir = win_data_dir
elif "Linux" in platform.system():
    data_dir = linux_data_dir
all_img_lab_path = os.path.join(data_dir, all_img_lab)
train_record_path = os.path.join(data_dir, train_records)
test_record_path = os.path.join(data_dir, test_records)


def _dump_data():
    ''' Save data to tfrecords file '''
    # Read data from *.npz file
    try:
        npzfile = np.load(all_img_lab_path)
    finally:
        print("*"*75 + "\n* Note: Please run read_tif.py with python27 and generate all_img_lab.npz *\n" + "*"*75)
    train_images = npzfile['train_images']
    train_labels = npzfile['train_labels']
    test_images = npzfile['test_images']
    # Open TF record writers
    train_writer = tf.python_io.TFRecordWriter(train_record_path)
    test_writer = tf.python_io.TFRecordWriter(test_record_path)
    # Write images and labels into record files
    for image, label in zip(train_images, train_labels):
        example = tf.train.Example(features=tf.train.Features(feature={
            "labels": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()])), 
            "images": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
        }))
        train_writer.write(example.SerializeToString())
    for image in test_images:
        example = tf.train.Example(features=tf.train.Features(feature={
            "images": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
        }))
        test_writer.write(example.SerializeToString())
    train_writer.close()
    test_writer.close()


def read_decode(filename_queue):
    '''  Reads and parses examples from ISBI Challenge data files
    ### Params:
        * filename_queuq - string: file name queue of tensorflow
    ### Return:
        * An object representing a single example, with the following fields:
            height: number of rows in the result (32)
            width: number of columns in the result (32)
            depth: number of color channels in the result (3)
            key: a scalar string Tensor describing the filename & record number for this example.
            label: a [height, width, depth] uint8 Tensor with the label data
            uint8image: a [height, width, depth] uint8 Tensor with the image data
    '''

    class ISBIRecord(object):
        pass
    result = ISBIRecord()

    result.height = IMG_SIZE
    result.width = IMG_SIZE
    result.depth = IMG_DEPTH

    # Read a record, getting filenames from the filename_queue.
    reader = tf.TFRecordReader()
    result.key, serialized_value = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_value, 
        features={
            "labels": tf.FixedLenFeature([], tf.string),
            "images": tf.FixedLenFeature([], tf.string)
        })
    # Convert from a string to a vector of uint8
    label_record_bytes = tf.decode_raw(features["labels"], tf.uint8)
    image_record_bytes = tf.decode_raw(features["images"], tf.uint8)
    # Reshape the image/label from [depth * height * width] to [depth, height, width].
    label_depth_major = tf.reshape(label_record_bytes, [result.depth, result.height, result.width])
    image_depth_major = tf.reshape(image_record_bytes, [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.label = tf.transpose(label_depth_major, [1, 2, 0])
    result.uint8image = tf.transpose(image_depth_major, [1, 2, 0])

    return result


def distorted_data_abandoned():
    ''' Data augmentation. Abandoned! 
        Label and image do not changed synchronously.
    '''
    if not os.path.exists(train_record_path) or not os.path.exists(test_record_path):
        _dump_data()
    filename_queue = tf.train.string_input_producer([train_record_path])
    read_input = read_decode(filename_queue)
    reshaped_label = tf.cast(read_input.label, tf.float32)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    # Randomly crop a [height, width] section of the i mage.
    distorted_label = tf.random_crop(reshaped_label, [IMG_SIZE, IMG_SIZE, IMG_DEPTH])
    distorted_image = tf.random_crop(reshaped_image, [IMG_SIZE, IMG_SIZE, IMG_DEPTH])
    
    # Randomly flip the image horizontally and vertically
    distorted_label = tf.image.random_flip_left_right(distorted_label)
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_label = tf.image.random_flip_up_down(distorted_label)
    distorted_image = tf.image.random_flip_up_down(distorted_image)

    # Randomly adjust the brightness and contrast
    distorted_label = tf.image.random_brightness(distorted_label, max_delta=63)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_label = tf.image.random_contrast(distorted_label, lower=0.2, upper=1.8)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_label = tf.image.per_image_standardization(distorted_label)
    float_image = tf.image.per_image_standardization(distorted_image)

    float_label.set_shape([IMG_SIZE, IMG_SIZE, 1])
    float_image.set_shape([IMG_SIZE, IMG_SIZE, 1])

    min_queue_examples = int(IMG_SLICES * 0.5)

    return _generate_image_and_label_batch(float_image, float_label, 
                                           min_queue_examples, BATCH_SIZE,
                                           shuffle=True)

def distorted_data():
    if not os.path.exists(train_record_path) or not os.path.exists(test_record_path):
        _dump_data()
    filename_queue = tf.train.string_input_producer([train_record_path])
    read_input = read_decode(filename_queue)
    reshaped_label = tf.cast(read_input.label, tf.float32)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    
    

