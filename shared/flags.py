import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_dir', '/home/vaibhav/tfprograms/dataset',
					'Directory containing images and tags.')

flags.DEFINE_string('tfrecords_dir', '/home/vaibhav/tfprograms/generated/tfrecords',
					'Directory to store TFRecords.')


flags.DEFINE_string('train_checkpoint_dir', '/home/vaibhav/tfprograms/generated/checkpoints/train',
					'Checkpoint directory to save training progress.')

flags.DEFINE_string('eval_checkpoint_dir', '/home/vaibhav/tfprograms/generated/checkpoints/eval',
					'Checkpoint directory to save training progress.')

flags.DEFINE_string('checkpoint_file', 'train.ckpt',
					'Checkpoint file to save training progress.')

flags.DEFINE_string('hashtags_output_file', '/home/vaibhav/tfprograms/generated/hashtag_map.txt',
					'Hashtags used for training.')

flags.DEFINE_integer('training_set_size', 100, 'Training set size.')
flags.DEFINE_integer('eval_set_size', 20, 'Evaluation set size.')

flags.DEFINE_integer('image_width', 299, 'Image width after cropping')
flags.DEFINE_integer('image_height', 299, 'Image height after cropping')

flags.DEFINE_integer('batch_size', 10, 'Images to process in 1 batch')

# Deliberately the same as image_width
flags.DEFINE_integer('label_set_size', 299, 'Number of labels in training/evaluation set')