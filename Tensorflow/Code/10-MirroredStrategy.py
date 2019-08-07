import argparse
import tensorflow as tf    # Assert tf.VERSION >= 1.13
from tensorflow import keras
from tensorflow.python.distribute import values
from tensorflow.python.util import nest
import numpy as np
import collections

# import os
# from tensorflow.python.util import deprecation
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
# deprecation._PRINT_DEPRECATION_WARNINGS = False
print(tf.VERSION, tf.keras.__version__)


def gen_data(bs):
    image = np.random.uniform(-1, 1, size=(bs, 2))
    label = image[:, 0] > 0
    return image.astype(np.float32), label.astype(np.int32)


def network(features, labels):
    out = keras.layers.Dense(2, kernel_initializer="ones")(features)
    losses = tf.losses.sparse_softmax_cross_entropy(logits=out, labels=labels,
                                                    reduction=tf.losses.Reduction.NONE)
    return losses


def solver(loss):
    global_step = tf.train.get_or_create_global_step()
    opt = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = opt.minimize(loss, global_step=global_step)
    return train_op


class Wrapper(collections.namedtuple('Father', ['train_op', 'losses', 'loss'])):
    def __new__(cls, train_op, losses, loss):
        return super(Wrapper, cls).__new__(cls, train_op, losses, loss)


def model_fn(features, labels):
    losses = network(features, labels)
    loss = tf.reduce_mean(losses)
    train_op = solver(loss)
    return Wrapper(train_op=train_op, losses=losses, loss=loss)


def per_device_dataset(batch, devices):
    index = {}

    def get_ith(i_):
        return lambda x: x[i_]
    
    for i, d in enumerate(devices):
        index[d] = nest.map_structure(get_ith(i), batch)
        
    return values.regroup(index)


def main_1():
    tf.reset_default_graph()
    tf.set_random_seed(1234)
    x_input = tf.placeholder(tf.float32, shape=(2, 2), name="x_input")
    y_input = tf.placeholder(tf.int32, shape=(2, ), name="y_input")
    wp = model_fn(x_input, y_input)

    np.random.seed(1234)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        x, y = gen_data(bs=2)
        _, l1, l2 = sess.run([wp.train_op, wp.loss, wp.losses], feed_dict={x_input: x, y_input: y})
        if i % 100 == 0:
            print("step {}, loss {} {}".format(i, l1, l2))


def main_2():
    tf.reset_default_graph()
    tf.set_random_seed(1234)
    strategy = tf.distribute.MirroredStrategy(["device:GPU:0", "device:GPU:1"])

    with strategy.scope():
        x_input = tf.placeholder(tf.float32, shape=(2, 2), name="x_input")
        y_input = tf.placeholder(tf.int32, shape=(2, ), name="y_input")

        features, labels = per_device_dataset((tf.reshape(x_input, (2, 1, 2)),
                                               tf.reshape(y_input, (2, 1))), strategy.extended._devices)
        grouped_wp = strategy.call_for_each_replica(model_fn, args=(features, labels))
        mean_loss = strategy.reduce(tf.distribute.get_loss_reduction(), grouped_wp.loss)
        concat_loss = tf.stack(strategy.unwrap(grouped_wp.loss), axis=0)
        train_op = strategy.group(grouped_wp.train_op)

        np.random.seed(1234)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            x, y = gen_data(bs=2)
            _, l1, l2 = sess.run([train_op, mean_loss, concat_loss], feed_dict={x_input: x, y_input: y})
            if i % 100 == 0:
                print("step {}, loss {} {}".format(i, l1, l2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=int)
    args = parser.parse_args()
    if args.run == 1:
        main_1()
    elif args.run == 2:
        main_2()
