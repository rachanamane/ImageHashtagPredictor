import tensorflow as tf

# Unused import - Required for flags - Don't remove
import shared.flags
FLAGS = tf.app.flags.FLAGS

pool_size = 2
padding = "same"


def logits(image, user_history, print_debug=True):
    current_tensor_width = FLAGS.image_width
    current_tensor_height = FLAGS.image_height
    if print_debug:
        print("Image shape %s" % image.shape)

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 299, 299, 3]
    # Output Tensor Shape: [batch_size, 299, 299, 32]
    kernel_size=5
    current_filters = 32
    conv1 = tf.layers.conv2d(
            inputs=image,
            filters=current_filters,
            kernel_size=[kernel_size, kernel_size],
            bias_initializer=tf.random_normal_initializer(stddev=0.1),
            padding=padding,
            activation=tf.nn.relu,
            name="my_Conv_layer_1")
    if print_debug:
        print("conv1 %s" % conv1.shape)
    current_tensor_width = current_tensor_width if padding == "same" else current_tensor_width - kernel_size + 1
    current_tensor_height = current_tensor_height if padding == "same" else current_tensor_height - kernel_size + 1

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 299, 299, 32]
    # Output Tensor Shape: [batch_size, 149, 149, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=pool_size, strides=pool_size, name="my_Pool_layer_1")
    current_tensor_width /= pool_size
    current_tensor_height /= pool_size
    if print_debug:
        print("pool1 %s" % pool1.shape)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 149, 149, 32]
    # Output Tensor Shape: [batch_size, 149, 149, 64]
    current_filters *= 2
    kernel_size=5
    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=current_filters,
          kernel_size=[kernel_size, kernel_size],
          bias_initializer=tf.random_normal_initializer(stddev=0.1),
          padding=padding,
          activation=tf.nn.relu,
          name="my_Conv_layer_2")
    current_tensor_width = current_tensor_width if padding == "same" else current_tensor_width - kernel_size + 1
    current_tensor_height = current_tensor_height if padding == "same" else current_tensor_height - kernel_size + 1
    if print_debug:
        print("conv2 %s" % conv2.shape)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 149, 149, 64]
    # Output Tensor Shape: [batch_size, 74, 74, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=pool_size, strides=pool_size, name="my_Pool_layer_2")
    current_tensor_width /= pool_size
    current_tensor_height /= pool_size
    if print_debug:
        print("pool2 %s" % pool2.shape)

    # Convolutional Layer #3
    # Computes 128 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 74, 74, 64]
    # Output Tensor Shape: [batch_size, 74, 74, 128]
    current_filters *= 2
    kernel_size=5
    conv3 = tf.layers.conv2d(
          inputs=pool2,
          filters=current_filters,
          kernel_size=[kernel_size, kernel_size],
          bias_initializer=tf.random_normal_initializer(stddev=0.1),
          padding=padding,
          activation=tf.nn.relu,
          name="my_Conv_layer_3")
    current_tensor_width = current_tensor_width if padding == "same" else current_tensor_width - kernel_size + 1
    current_tensor_height = current_tensor_height if padding == "same" else current_tensor_height - kernel_size + 1
    if print_debug:
        print("conv3 %s" % conv3.shape)

    # Pooling Layer #3
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 74, 74, 128]
    # Output Tensor Shape: [batch_size, 37, 37, 128]
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=pool_size, strides=pool_size, name="my_Pool_layer_3")
    current_tensor_width /= pool_size
    current_tensor_height /= pool_size
    if print_debug:
        print("pool3 %s" % pool3.shape)

    # [-1, current_tensor_width * current_tensor_height * current_filters],
    pool_flat = tf.layers.flatten(pool3, name="my_Pool_layer_flat")
    if print_debug:
        print("pool_flat %s" % pool_flat.shape)
        print("user_history %s" % user_history.shape)

    weighted_user_history = tf.multiply(tf.constant(1.0, shape=user_history.shape), user_history)

    pool_flat_with_history = tf.concat([pool_flat, weighted_user_history], axis=1, name="my_user_history_concat_layer")
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


def loss(logits, labels):
    losses = tf.losses.sigmoid_cross_entropy(labels, logits)
    return losses
