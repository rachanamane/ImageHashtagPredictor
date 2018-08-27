import tensorflow as tf

# Unused import - Required for flags - Don't remove
import shared.flags
FLAGS = tf.app.flags.FLAGS

pool_size = 2

def logits(image):
    current_tensor_width = FLAGS.image_width
    current_tensor_height = FLAGS.image_height
    print(image.shape)

    # Convolutional Layer #1
    # Computes 16 features using a 6x6 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 299, 299, 3]
    # Output Tensor Shape: [batch_size, 294, 294, 32]
    current_filters = 16
    conv1 = tf.layers.conv2d(
            inputs=image,
            filters=current_filters,
            kernel_size=[6, 6],
            activation=tf.nn.relu)
    print("conv1 %s" % conv1.shape)
    current_tensor_width = current_tensor_width - 6 + 1
    current_tensor_height = current_tensor_height - 6 + 1

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=pool_size, strides=pool_size)
    current_tensor_width /= pool_size
    current_tensor_height /= pool_size
    print("pool1 %s" % pool1.shape)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 16]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    current_filters *= 2
    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=current_filters,
          kernel_size=[5, 5],
          activation=tf.nn.relu)
    current_tensor_width = current_tensor_width - 5 + 1
    current_tensor_height = current_tensor_height - 5 + 1
    print("conv2 %s" % conv2.shape)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=pool_size, strides=pool_size)
    current_tensor_width /= pool_size
    current_tensor_height /= pool_size
    print("pool2 %s" % pool2.shape)

    pool2_flat = tf.reshape(pool2, [-1, current_tensor_width * current_tensor_height * current_filters])
    print("pool2_flat %s" % pool2_flat.shape)

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense1 = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)
    print("dense1 %s" % dense1.shape)

    # TODO: Consider adding dropout, to remove labels x% of least probable labels

    logits = tf.layers.dense(inputs=dense1, units=FLAGS.label_set_size)

    return tf.nn.sigmoid(logits)


def loss(logits, labels):
    losses = tf.losses.sigmoid_cross_entropy(labels, logits)
    return losses
