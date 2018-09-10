import tensorflow as tf

# Unused import - Required for flags - Don't remove
import shared.flags
FLAGS = tf.app.flags.FLAGS

default_pool_size = 2
default_padding = "same"


def run_conv(inputs, filters, kernel_size, name=''):
    if not name:
        name = "my_conv_layer_%sx%s" % (filters, kernel_size)
    layer = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            bias_initializer=tf.random_normal_initializer(stddev=0.1),
            padding=default_padding,
            activation=tf.nn.relu,
            name=name)
    print("%s: %s" % (name, layer.shape))
    return layer

def run_conv_without_relu(inputs, filters, kernel_size, name=''):
    if not name:
        name = "my_conv_layer_%sx%s" % (filters, kernel_size)
    layer = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            bias_initializer=tf.random_normal_initializer(stddev=0.1),
            padding=default_padding,
            activation=None,
            name=name)
    print("%s: %s" % (name, layer.shape))
    return layer


def run_pool(inputs, name, pool_size=default_pool_size, strides=default_pool_size, padding='valid'):
    layer = tf.layers.max_pooling2d(inputs=inputs, pool_size=pool_size, strides=strides, name=name, padding=padding)
    print("%s: %s" % (name, layer.shape))
    return layer


def run_initial_conv(image):
    conv1 = run_conv(image, filters=64, kernel_size=7)
    pool1 = run_pool(conv1, name="my_pool_1")

    conv2 = run_conv(pool1, filters=192, kernel_size=3)
    pool2 = run_pool(conv2, name="my_pool_2")
    return pool2


def run_parallel_conv(inputs):
    # Try filters=16 everywhere
    # Conv layer 1x1
    conv1x1 = run_conv_without_relu(inputs, filters=32, kernel_size=1, name="conv_for_1x1")

    # Conv layer 3x3
    conv1x1_reduce_for_3 = run_conv(inputs, filters=16, kernel_size=1, name="reduce_for_3x3")
    conv3x3 = run_conv_without_relu(conv1x1_reduce_for_3, filters=64, kernel_size=3, name="conv_for_3x3")

    # Conv layer 5x5
    conv1x1_reduce_for_5 = run_conv(inputs, filters=16, kernel_size=1, name="reduce_for_5x5")
    conv5x5 = run_conv_without_relu(conv1x1_reduce_for_5, filters=32, kernel_size=5, name="conv_for_5x5")

    max_pool = run_pool(inputs, name="pool_parallel_3x3", pool_size=3, strides=1, padding='same')
    conv_after_pool = run_conv_without_relu(max_pool, filters=32, kernel_size=1, name="conv_after_max_pool")

    return tf.concat([conv1x1, conv3x3, conv5x5, conv_after_pool], axis=3, name="concat_parallel_layers")



def parallel_conv2d_logits(image, user_history, print_debug=True):
    if print_debug:
        print("Image shape %s" % image.shape)

    initial_conv = run_initial_conv(image)

    flattened_conv = run_parallel_conv(initial_conv)

    # [-1, current_tensor_width * current_tensor_height * current_filters],
    pool_flat = tf.layers.flatten(flattened_conv, name="my_Pool_layer_flat")
    if print_debug:
        print("pool_flat %s" % pool_flat.shape)
        print("user_history %s" % user_history.shape)

    weighted_user_history = tf.multiply(tf.constant(1.0, shape=user_history.shape), user_history)

    pool_flat_with_history = tf.nn.relu(tf.concat([pool_flat, weighted_user_history], axis=1, name="my_user_history_concat_layer"))
    if print_debug:
        print("pool_flat_with_history %s" % pool_flat_with_history.shape)

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 74 * 74 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense1 = tf.layers.dense(inputs=pool_flat_with_history, units=1024, activation=tf.nn.relu, name="my_Dense_layer_1")
    if print_debug:
        print("dense1 %s" % dense1.shape)

    # TODO: Consider adding dropout, to remove labels x% of least probable labels

    logits = tf.layers.dense(inputs=dense1, units=FLAGS.label_set_size, name="my_Logits_layer")

    return logits
