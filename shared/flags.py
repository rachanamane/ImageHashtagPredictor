import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_dir', '/Users/namitr/tfprograms/dataset',
					'Directory containing images and tags.')

flags.DEFINE_string('tfrecords_dir', '/Users/namitr/tfprograms/generated/tfrecords',
					'Directory to store TFRecords.')

flags.DEFINE_string('checkpoint_file', '/Users/namitr/tfprograms/generated/checkpoint/train.ckpt',
					'Checkpoint file to save training progress.')


flags.DEFINE_integer('training_set_size', 100, 'Training set size.')

flags.DEFINE_integer('image_width', 224, 'Image width after cropping')
flags.DEFINE_integer('image_height', 224, 'Image height after cropping')