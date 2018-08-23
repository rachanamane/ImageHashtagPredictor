import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_dir', '/Users/namitr/tfprograms/dataset/dogsofinstagram',
					'Directory containing images and tags.')

flags.DEFINE_string('tfrecords_dir', '/Users/namitr/tfprograms/generated/tfrecords',
					'Directory to store TFRecords.')

flags.DEFINE_string('checkpoint_file', '/Users/namitr/tfprograms/generated/checkpoint/train.ckpt',
					'Checkpoint file to save training progress.')

flags.DEFINE_string('hashtags_output_file', '/Users/namitr/tfprograms/generated/hashtag_map.txt',
					'Hashtags used for training.')

flags.DEFINE_integer('training_set_size', 1000, 'Training set size.')
flags.DEFINE_integer('eval_set_size', 100, 'Evaluation set size.')

flags.DEFINE_integer('image_width', 299, 'Image width after cropping')
flags.DEFINE_integer('image_height', 299, 'Image height after cropping')

flags.DEFINE_integer('batch_size', 10, 'Images to process in 1 batch')

# TODO: Remove this and allow variable labels per image
flags.DEFINE_integer('labels_per_image', 10, 'Fixed number of labels per image')

# Deliberately the same as image_width
flags.DEFINE_integer('label_set_size', 299, 'Number of labels in training/evaluation set')