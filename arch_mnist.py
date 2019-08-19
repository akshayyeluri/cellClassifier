import tensorflow as tf
import numpy as np

WIDTH = 28
HEIGHT = 28
DEPTH = 1
SEED = 42


def reshape_data(data):
    return data.reshape(-1, WIDTH, HEIGHT, DEPTH)


def make_vars_on_cpu(NCLASS):
    with tf.compat.v1.variable_scope(__name__):
        tf.compat.v1.get_variable(name="conv1_w", \
          initializer=tf.random.truncated_normal([5, 5, DEPTH, 32], stddev=0.1, seed=SEED))
        tf.compat.v1.get_variable(name="conv1_b", initializer=tf.zeros([32]))
        
        tf.compat.v1.get_variable(name="conv2_w", \
          initializer=tf.random.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=SEED))
        tf.compat.v1.get_variable(name="conv2_b", initializer=tf.constant(0.1, shape=[64]))
        
        tf.compat.v1.get_variable(name="fc1_w", \
          initializer=tf.random.truncated_normal([7 * 7 * 64, 512], stddev=0.1, seed=SEED))
        tf.compat.v1.get_variable(name="fc1_b", initializer=tf.constant(0.1, shape=[512]))
        
        tf.compat.v1.get_variable(name="fc2_w", \
          initializer=tf.random.truncated_normal([512, NCLASS], stddev=0.1, seed=SEED))
        tf.compat.v1.get_variable(name="fc2_b", initializer=tf.constant(0.1, shape=[NCLASS]))


def model(inp, train=False):
    # Get the variables
    with tf.compat.v1.variable_scope(__name__, reuse=True):
        conv1_weights = tf.compat.v1.get_variable("conv1_w")
        conv1_biases = tf.compat.v1.get_variable("conv1_b")
        conv2_weights = tf.compat.v1.get_variable("conv2_w")
        conv2_biases = tf.compat.v1.get_variable("conv2_b")
        fc1_weights = tf.compat.v1.get_variable("fc1_w")
        fc1_biases = tf.compat.v1.get_variable("fc1_b")
        fc2_weights = tf.compat.v1.get_variable("fc2_w")
        fc2_biases = tf.compat.v1.get_variable("fc2_b")

    # Wire them into a convolutional neural net
    conv1 = tf.nn.conv2d(inp, conv1_weights, strides=[1,1,1,1], padding="SAME")
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], \
                           strides=[1, 2, 2, 1], padding="VALID")
    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1], padding="SAME")
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], \
                           strides=[1, 2, 2, 1], padding="VALID")
    size = pool2.get_shape().as_list()
    pool2_flat = tf.reshape(pool2, [size[0], size[1] * size[2] * size[3]])
    dense1 = tf.nn.relu(tf.matmul(pool2_flat, fc1_weights) + fc1_biases)
    if train:
        # Add L2 regularization and dropout
        dense1 = tf.nn.dropout(dense1, rate=0.5, seed=SEED)
        regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases)+\
                        tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
        regularizers = 5e-4 * regularizers
        
    dense2 = tf.nn.bias_add(tf.matmul(dense1, fc2_weights), fc2_biases)
    return (dense2, regularizers) if train else dense2
