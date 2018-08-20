import tensorflow as tf

# Unused import - Required for flags - Don't remove
import shared.flags
FLAGS = tf.app.flags.FLAGS

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.02, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def flower_inference(image_batch):
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(image_batch, [-1, FLAGS.image_width, FLAGS.image_height, 3])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1) # 112

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2) # 56

    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3) # 28

    W_conv4 = weight_variable([5, 5, 128, 256])
    b_conv4 = bias_variable([256])

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4) # 14

    W_conv5 = weight_variable([5, 5, 256, 256])
    b_conv5 = bias_variable([256])

    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    h_pool5 = max_pool_2x2(h_conv5) # 7

    W_fc1 = weight_variable([7*7*256, 2048])
    b_fc1 = bias_variable([2048])

    h_pool5_flat = tf.reshape(h_pool5, [-1, 7*7*256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, 1.0)

    W_fc2 = weight_variable([2048, 256])
    b_fc2 = bias_variable([256])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    W_fc3 = weight_variable([256, 64])
    b_fc3 = bias_variable([64])

    h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    W_fc4 = weight_variable([64, 5])
    b_fc4 = bias_variable([5])

    y_conv = tf.nn.softmax(tf.matmul(h_fc3, W_fc4) + b_fc4)
#    y_conv = tf.matmul(h_fc3, W_fc4) + b_fc4

    return y_conv