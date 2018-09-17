import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_dir', '/home/vaibhav/tfprograms/dataset',
					'Directory containing images and tags.')

flags.DEFINE_string('tfrecords_dir', '/home/vaibhav/tfprograms/generated/tfrecords',
					'Directory to store TFRecords.')

flags.DEFINE_string('train_checkpoint_dir', '/home/vaibhav/tfprograms/generated/checkpoints',
					'Checkpoint directory to save training progress.')

flags.DEFINE_string('checkpoint_file', 'train',
					'Checkpoint file to save training progress.')

flags.DEFINE_string('tensorboard_logs_dir', '/home/vaibhav/tfprograms/generated/tensorboard',
					'Checkpoint directory to save training progress.')

flags.DEFINE_string('hashtags_output_file', '/home/vaibhav/tfprograms/generated/hashtag_map.txt',
					'Hashtags used for training.')

flags.DEFINE_string('user_history_output_file', '/home/vaibhav/tfprograms/generated/user_history.txt',
					'User hashtag usage history.')

flags.DEFINE_integer('num_epochs', 6, 'Number of epochs')

flags.DEFINE_integer('image_width', 299, 'Image width after cropping')
flags.DEFINE_integer('image_height', 299, 'Image height after cropping')

flags.DEFINE_integer('batch_size', 25, 'Images to process in 1 batch')

flags.DEFINE_integer('label_set_size', 32, 'Number of labels in training/evaluation set')

flags.DEFINE_integer('images_per_shard', 500, 'Images that are written in one TFRecord shard')

flags.DEFINE_integer('train_write_shards', 84, 'Number of shards for creating TF Records for training')
flags.DEFINE_integer('eval_write_shards', 14, 'Number of shards for creating TF Records for evaluation')

flags.DEFINE_integer('model_train_shards', 84, 'Number of shards for training data')
flags.DEFINE_integer('model_eval_shards', 2, 'Number of shards for evaluation data')

flags.DEFINE_integer('eval_checkpoint_epoch', 3,
					 'If -1, run evaluation against all epochs. Otherwise, run evaluation against the provided epoch')
