from shared.features import ImageHashtagFeatures
from shared.singleimageobject import SingleImageObject

import os
import tensorflow as tf

from os import listdir

# Unused import - Required for flags - Don't remove
import shared.flags
FLAGS = tf.app.flags.FLAGS


def read_image_from_tfrecord(filename_queue):
    # TODO: https://www.programcreek.com/python/example/90543/tensorflow.TFRecordReader
    # There are more examples of reading TFRecords in batch, try that later.
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features_dict = tf.parse_single_example(serialized_example, features={
        ImageHashtagFeatures.heightFeature: tf.FixedLenFeature([], tf.int64),
        ImageHashtagFeatures.widthFeature: tf.FixedLenFeature([], tf.int64),
        ImageHashtagFeatures.imageRawFeature: tf.FixedLenFeature([], tf.string),
        ImageHashtagFeatures.labelsFeature: tf.VarLenFeature(tf.int64),
        ImageHashtagFeatures.encodedLabelsFeature: tf.VarLenFeature([FLAGS.label_set_size], tf.int64),
    })
    return SingleImageObject(features_dict)


def get_tfrecord_filenames():
    files = listdir(FLAGS.tfrecords_dir)
    return [os.path.join(FLAGS.tfrecords_dir, x) for x in files if x.endswith(".tfrecord")]


def read_tf_records():
    filename_queue = tf.train.string_input_producer(get_tfrecord_filenames())
    image_object = read_image_from_tfrecord(filename_queue)
    # TODO: Add batch size flag
    batch_image, batch_labels, batch_encoded_labels = tf.train.batch(
            [image_object.image_raw, image_object.labels, image_object.encoded_labels],
            batch_size=FLAGS.batch_size,
            num_threads=1)

    return batch_image, batch_labels
