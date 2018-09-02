import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_dir', '/home/vaibhav/tfprograms/dataset',
					'Directory containing images and tags.')

flags.DEFINE_string('tfrecords_dir', '/home/vaibhav/tfprograms/generated/tfrecords',
					'Directory to store TFRecords.')

flags.DEFINE_string('train_checkpoint_dir', '/home/vaibhav/tfprograms/generated/checkpoints',
					'Checkpoint directory to save training progress.')

flags.DEFINE_string('checkpoint_file', 'train.ckpt',
					'Checkpoint file to save training progress.')

flags.DEFINE_string('tensorboard_logs_dir', '/home/vaibhav/tfprograms/generated/tensorboard',
					'Checkpoint directory to save training progress.')

flags.DEFINE_string('hashtags_output_file', '/home/vaibhav/tfprograms/generated/hashtag_map.txt',
					'Hashtags used for training.')

flags.DEFINE_integer('num_epochs', 3, 'Number of epochs')

flags.DEFINE_integer('training_set_size', 42000, 'Training set size.')
flags.DEFINE_integer('eval_set_size', 7000, 'Evaluation set size.')

flags.DEFINE_integer('image_width', 299, 'Image width after cropping')
flags.DEFINE_integer('image_height', 299, 'Image height after cropping')

flags.DEFINE_integer('batch_size', 25, 'Images to process in 1 batch')

flags.DEFINE_integer('label_set_size', 32, 'Number of labels in training/evaluation set')

flags.DEFINE_integer('train_write_shards', 80, 'Number of shards for training data')
flags.DEFINE_integer('eval_write_shards', 40, 'Number of shards for training data')

flags.DEFINE_boolean('use_insta_posts_with_multiple_images', False,
					 'Use posts that have multiple images in a single post')
