import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_string('train_directory', '/Users/namitr/tfprograms/dataset/#yummy',
                           'Training data directory')

if __name__ == "__main__":
    print(FLAGS.train_directory)