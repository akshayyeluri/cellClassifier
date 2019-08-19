import tensorflow as tf
import numpy as np

WIDTH = 32738
HEIGHT = 1
DEPTH = 1
SEED = 42


def reshape_data(data):
    return data.reshape(-1, WIDTH, HEIGHT, DEPTH)


def make_vars_on_cpu(NCLASS):
    with tf.compat.v1.variable_scope(__name__):
        tf.compat.v1.get_variable(name="conv0_w", \
          initializer=tf.random.truncated_normal([3, 1, DEPTH, 32], stddev=0.1, seed=SEED))
        tf.compat.v1.get_variable(name="conv0_b", initializer=tf.zeros([32]))
        
        tf.compat.v1.get_variable(name="conv1_w", \
          initializer=tf.random.truncated_normal([3, 1, 32, 64], stddev=0.1, seed=SEED))
        tf.compat.v1.get_variable(name="conv1_b", initializer=tf.constant(0.1, shape=[64]))
        
        tf.compat.v1.get_variable(name="conv2_w", \
          initializer=tf.random.truncated_normal([2, 1, 64, 128], stddev=0.1, seed=SEED))
        tf.compat.v1.get_variable(name="conv2_b", initializer=tf.constant(0.1, shape=[128]))

        tf.compat.v1.get_variable(name="fc1_w", \
          initializer=tf.random.truncated_normal([34 * 128, 512], stddev=0.1, seed=SEED))
        tf.compat.v1.get_variable(name="fc1_b", initializer=tf.constant(0.1, shape=[512]))
        
        tf.compat.v1.get_variable(name="fc2_w", \
          initializer=tf.random.truncated_normal([512, NCLASS], stddev=0.1, seed=SEED))
        tf.compat.v1.get_variable(name="fc2_b", initializer=tf.constant(0.1, shape=[NCLASS]))


def model(inp, train=False):
    # Params
    NCONV = 3
    pads = ["VALID", "SAME", "VALID"]
    pool_ks = [6, 16, 10]

    # Get the variables
    conv_ws, conv_bs = [], []
    with tf.compat.v1.variable_scope(__name__, reuse=True):
        for i in range(NCONV):
            conv_ws.append(tf.compat.v1.get_variable(f"conv{i}_w"))
            conv_bs.append(tf.compat.v1.get_variable(f"conv{i}_b"))
        fc1_weights = tf.compat.v1.get_variable("fc1_w")
        fc1_biases = tf.compat.v1.get_variable("fc1_b")
        fc2_weights = tf.compat.v1.get_variable("fc2_w")
        fc2_biases = tf.compat.v1.get_variable("fc2_b")

    # Wire them into a convolutional neural net
    pool = inp
    for i in range(NCONV):
        conv = tf.nn.conv2d(pool, conv_ws[i], strides=[1,1,1,1], padding=pads[i])
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv_bs[i]))
        pool = tf.nn.max_pool(relu, ksize=[1, pool_ks[i], 1, 1], \
                           strides=[1, pool_ks[i], 1, 1], padding="VALID")

    size = pool.get_shape().as_list()
    pool_flat = tf.reshape(pool, [size[0], size[1] * size[2] * size[3]])
    dense1 = tf.nn.relu(tf.matmul(pool_flat, fc1_weights) + fc1_biases)
    if train:
        # Add L2 regularization and dropout
        dense1 = tf.nn.dropout(dense1, rate=0.5, seed=SEED)
        regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases)+\
                        tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
        regularizers = 5e-4 * regularizers
        
    dense2 = tf.nn.bias_add(tf.matmul(dense1, fc2_weights), fc2_biases)
    return (dense2, regularizers) if train else dense2
