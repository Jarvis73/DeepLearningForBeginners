#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
UNet implementation

@author: Jarvis ZHANG
@date: 2017/9/2
@framework: Tensorflow
@editor: VS Code
"""

import os
import cv2
import glob
import platform
import numpy as np
import pandas as pd
import tensorflow as tf

from skimage.morphology import dilation, square
from skimage.feature import canny

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# image size
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 520
IMAGE_DEPTH = 3

# data sets info
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 85
NUM_EXAMPLES_PER_EPOCH_FOR_TEST  = 80

# number of preprocess threads
NUM_PREPROCESS_THREADS = 2

# path
WINDOWS_DATA_DIR = "C:\\DataSet\\MICCAI2015\\Warwick_QU_Dataset"
LINUX_DATA_DIR = "/home/jarvis/DataSet/Warwick_QU_Dataset"

if "Windows" in platform.system():
    WQU_ROOT_DIR = WINDOWS_DATA_DIR
elif "Linux" in platform.system():
    WQU_ROOT_DIR = LINUX_DATA_DIR

WQU_TRAIN_DIR = os.path.join(WQU_ROOT_DIR, "train")
WQU_TEST_DIR = os.path.join(WQU_ROOT_DIR, "test")
TF_TRAIN_DIR = os.path.join(WQU_ROOT_DIR, "generate_train")
TF_CROPED_TRAIN_DIR = os.path.join(WQU_ROOT_DIR, "croped_train")

def split_train_and_test_csv():
    """ Split the Grade.csv into two csv -- train.csv and test.csv 
    """
    df = pd.read_csv(os.path.join(WQU_ROOT_DIR, "Grade.csv"))
    df["contour"] = [os.path.join(TF_CROPED_TRAIN_DIR, name + "_contour.png") for name in df["name"]]
    df["annotation"] = [os.path.join(TF_CROPED_TRAIN_DIR, name + "_anno.png") for name in df["name"]]
    df["name"] = [os.path.join(TF_CROPED_TRAIN_DIR, name + "_src.png") for name in df["name"]]
    train_info = df.iloc[80:, :]
    test_info = df.iloc[:80, :]

    train_info.to_csv(os.path.join(WQU_ROOT_DIR, "train.csv"), index=False)
    test_info.to_csv(os.path.join(WQU_ROOT_DIR, "test.csv"), index=False)


def create_contour_from_anno(data_dir, process_only_one=None):
    """ Generate contour images from annotation images.

    ### Args:
        * `data_dir`: WQU_TRAIN_DIR or WQU_TEST_DIR
        * `process_only_one`: Name of one image to test this function
    """
    if process_only_one is not None:
        anno_list = [process_only_one]
    else:
        anno_list = glob.glob(os.path.join(data_dir, "*anno.bmp"))
    
    dst_dir = TF_TRAIN_DIR
    
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for one_image in anno_list:
        one_image = os.path.basename(one_image)
        image = cv2.imread(os.path.join(data_dir, one_image), cv2.IMREAD_GRAYSCALE)
        image[image > 0] = 255
        contour = canny(image)
        contour = dilation(contour, square(3)).astype(np.uint8) * 255
        print(one_image)
        cv2.imwrite(os.path.join(dst_dir, one_image[:-9] + "_anno.png"), image)
        cv2.imwrite(os.path.join(dst_dir, one_image[:-9] + "_contour.png"), contour)
        origin = cv2.imread(os.path.join(data_dir, one_image[:-9] + ".bmp"), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(os.path.join(dst_dir, one_image[:-9] + "_src.png"), origin)


def crop_and_generate_dataset():
    """ Crop the original image with size of [520, 768]
    """
    filepath = TF_TRAIN_DIR + "*.png"
    filelist = glob.glob(filepath)
    dst_dir = TF_CROPED_TRAIN_DIR
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    
    for name in filelist:
        print(name)
        image = cv2.imread(name)
        if image.shape[0] < 522:
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        elif image.shape[0] == 522:
            image = image[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
        newname = os.path.join(dst_dir, os.path.basename(name))
        cv2.imwrite(newname, image)
        


def read_from_queue(path, channels, dtype=tf.uint8):
    return tf.image.decode_png(path, channels=channels, dtype=dtype)


def read_wqu(all_files_queue):
    """Reads and parses examples from Warwick QU data files.
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
    class WQURecord(object):
        pass
    
    result = WQURecord()

    # Read a record, getting filenames from the filename_queue.
    text_reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_content = text_reader.read(all_files_queue)
    s_path, _, _, _, c_path, a_path = tf.decode_csv(csv_content, 
                                                record_defaults=[[""]] * 6)

    result.uint8image = read_from_queue(tf.read_file(s_path), channels=3)
    contour = read_from_queue(tf.read_file(c_path), channels=1)
    segment = read_from_queue(tf.read_file(a_path), channels=1)
    result.label = tf.concat([contour, segment], 2)

    return result


def get_read_input(eval_data=False):
    """ Fetch input data row by row from CSV files.
    ### Args:
        * `eval_data`: Bool representing whether to read from train or test directories.
    ### Returns:
        * `read_input`: An object representing a single example.
        * `reshaped_image`: Image of type tf.float32, reshaped to correct dimensions.
    """
    # Create queues that produce the filenames and labels to read.
    pref = "test" if eval_data else "train"
    all_files_queue = tf.train.string_input_producer([os.path.join(WQU_ROOT_DIR, pref + '.csv')])

    # Read examples from files in the filename queue.
    read_input = read_wqu(all_files_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    read_input.label = tf.cast(read_input.label, tf.int32)

    return read_input, reshaped_image


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    ### Args:
        * `image`: 3-D Tensor of [height, width, depth] of type.float32.
        * `label`: 3-D Tensor of [height, width, 1] of type.int32.
        * `min_queue_examples`: int32, minimum number of samples to retain
            in the queue that provides of batches of examples.
        * `batch_size`: Number of images per batch.
        * `shuffle`: boolean indicating whether to use a shuffling queue.
    ### Returns:
        * `images`: Images. 4D tensor of [batch_size, height, width, depth] size.
        * `labels`: Labels. 4D tensor of [batch_size, height, width, 2] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = NUM_PREPROCESS_THREADS
    if shuffle:
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, labels


def inputs(eval_data, batch_size):
    """Construct input for Warwick QU evaluation using the Reader ops.
    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH] size.
        labels: Labels. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 2] size.
    """
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    read_input, reshaped_image = get_read_input(eval_data)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(reshaped_image)

    # Set the shapes of tensors.
    float_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
    read_input.label.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 2])

    # Set max intensity to 1
    read_input.label = tf.cast(tf.divide(read_input.label, 255), tf.int32)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue) # 85 * 0.4 = 34

    # Generate a batch of images and labels by building up a queue of examples.
    shuffle = False if eval_data else True
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=shuffle)

def distorted_input(eval_data, batch_size):
    pass


if __name__ == '__main__':
    # Split Grades.csv into train.csv and test.csv
    if True:
        split_train_and_test_csv()

    if False:
        process_only_one = "train_1_anno.bmp"
        create_contour_from_anno(WQU_TRAIN_DIR, process_only_one=None)
        create_contour_from_anno(WQU_TEST_DIR, process_only_one=None)

    if True:
        crop_and_generate_dataset()