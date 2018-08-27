import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

# Unused import - Required for flags - Don't remove
import shared.flags

FLAGS = tf.app.flags.FLAGS


def logits(image):
    logits, end_points = inception.inception_v3(
        image,
        num_classes=FLAGS.label_set_size,
        is_training=True)
    # TODO: Check for softmax function to get logits
   # logits = tf.sigmoid(logits)
    return logits


def loss(logits, labels):
    losses = tf.losses.sigmoid_cross_entropy(labels, logits)
    return losses
