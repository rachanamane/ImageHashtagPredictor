import tensorflow as tf
from tensorflow.contrib.slim.nets import inception


def _read_and_recode(filename_queue):

    """Read the TFRecords, process and return the tensor"""

    reader = tf.TFRecordReader()
    _, serialized_example =reader.read(filename_queue)
    # Returned a tuple of string scalar tensor
    features = tf.parse_single_example(serialized_example,
                                       features={'height': tf.FixedLenFeature([], tf.int64),
                                                 'width': tf.FixedLenFeature([], tf.int64),
                                                 'labels': tf.VarLenFeature(tf.int64),
                                                 'image_raw': tf.FixedLenFeature([], tf.string)})

