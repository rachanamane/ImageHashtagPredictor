import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_dir', '/Users/namitr/tfprograms/dataset',
					'Directory containing images and tags.')

flags.DEFINE_string('tfrecords_dir', '/Users/namitr/tfprograms/generated/tfrecords',
					'Directory to store TFRecords.')

flags.DEFINE_string('train_checkpoint_dir', '/Users/namitr/tfprograms/generated/checkpoints/train',
					'Checkpoint directory to save training progress.')

flags.DEFINE_string('eval_checkpoint_dir', '/Users/namitr/tfprograms/generated/checkpoints/eval',
					'Checkpoint directory to save training progress.')

flags.DEFINE_string('checkpoint_file', 'train.ckpt',
					'Checkpoint file to save training progress.')

flags.DEFINE_string('hashtags_output_file', '/Users/namitr/tfprograms/generated/hashtag_map.txt',
					'Hashtags used for training.')

flags.DEFINE_integer('training_set_size', 68000, 'Training set size.')
flags.DEFINE_integer('eval_set_size', 13000, 'Evaluation set size.')

flags.DEFINE_integer('image_width', 299, 'Image width after cropping')
flags.DEFINE_integer('image_height', 299, 'Image height after cropping')

flags.DEFINE_integer('batch_size', 25, 'Images to process in 1 batch')

flags.DEFINE_integer('label_set_size', 100, 'Number of labels in training/evaluation set')

flags.DEFINE_integer('train_write_shards', 80, 'Number of shards for training data')
flags.DEFINE_integer('eval_write_shards', 52, 'Number of shards for training data')

flags.DEFINE_boolean('use_insta_posts_with_multiple_images', True,
					 'Use posts that have multiple images in a single post')
