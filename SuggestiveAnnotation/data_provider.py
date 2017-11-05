#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
DCAN training procedure

@author: Jarvis ZHANG
@date: 2017/11/5
@framework: Tensorflow
@editor: VS Code
"""
import ntpath
import allPath
import tensorflow as tf

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_THICK = 32
IMAGE_DEEPTH = 1
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1200
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 150
NUM_PREPROCESS_THREADS = 16

def read_from_queue(path, channels, dtype=tf.uint8):
    return tf.image.decode_png(path, channels=channels, dtype=dtype)


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
    class LunaRecord(object):
        pass
    
    result = LunaRecord()

    # Read a record, getting filenames from the filename_queue.
    text_reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_content = text_reader.read(all_files_queue)
    o_path, m_path = tf.decode_csv(csv_content, record_defaults=[[""]] * 2)

    result.uint8image = read_from_queue(tf.read_file(o_path), channels=1)
    result.label = read_from_queue(tf.read_file(m_path), channels=1)

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

    all_files_queue = tf.train.string_input_producer([ntpath.join(allPath.SA_ROOT_DIR, pref + '.csv')])

    read_input = read_data_from_disk(all_files_queue)
    image = tf.cast(read_input.uint8image, tf.float32)
    label = tf.cast(read_input.label, tf.int32)

    return label, image


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


def input(eval_data, batch_size):
    """Construct input using the Reader ops.
    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, IMAGE_THICK, 1] size.
        labels: Labels. 4D tensor of [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_THICK 1] size.
    """
    if not eval_data:
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    
    label, image = get_read_input(eval_data)

    # Divide the center 32x32x32
    image = tf.image.central_crop(image, 0.5)
    label = tf.image.central_crop(label, 0.5)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(image)

    # Set max intensity to 1
    int_label = tf.cast(tf.divide(label, 255), tf.int32)

    # Set the shapes of tensors.
    float_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_THICK, IMAGE_DEEPTH])
    int_label.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_THICK, IMAGE_DEEPTH])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    shuffle = False if eval_data else True
    return _generate_image_and_label_batch(float_image, int_label,
                                           min_queue_examples, batch_size,
                                           shuffle=shuffle)


def distorted_input(eval_data, batch_size):
    pass

