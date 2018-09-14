import numpy as np
import tensorflow as tf
import preprocess.createHashtagsFile as createHashtagsFile

from model import createmodel
from os.path import isfile, join

# Unused import - Required for flags - Don't remove
import shared.flags
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('image_path', '/home/vaibhav/tfprograms/dataset/spaghetti_carbonara/42554.jpg',
					'Path of image to predict labels.')
flags.DEFINE_integer('predictions_count', 3,
					'Number of predictions.')


def _get_top_predictions(logits, k):
    _, indices = tf.nn.top_k(logits, k)
    return indices


def _read_metadata_file(hash_tag_filepath):
    hash_tags = []
    with open(hash_tag_filepath, 'r') as f:
        current_hashtags = f.readline().strip().split(",")
        for current_hashtag in current_hashtags:
            current_hashtag = current_hashtag.lower()
            if current_hashtag:
                hash_tags.append(current_hashtag)
        user_id = f.readline().strip()
    return sorted(list(set(hash_tags))), int(user_id)


def _get_read_image_metadata(file_path):
    if isfile(file_path) and file_path.endswith(".jpg"):
        hash_tag_filepath = file_path[0:-4] + ".txt"
        hash_tag_filepath_multiple_images = file_path[0:-6] + ".txt"
        if isfile(hash_tag_filepath):
            return _read_metadata_file(hash_tag_filepath)
        elif isfile(hash_tag_filepath_multiple_images):
            return _read_metadata_file(hash_tag_filepath_multiple_images)
        else:
            return [], 0
    else:
        raise Exception("Invalid file %s" % file_path)


def _read_user_history():
    user_history = []
    with open(FLAGS.user_history_output_file, 'r') as f:
        for i in range(10):
            curr_user_history = []
            cur_line_split = f.readline().strip().split(" ")
            for single_history in cur_line_split:
                curr_user_history.append(int(single_history))
            user_history.append(curr_user_history)
    return user_history


def predict_model(image_data, hashtag_name_lookup, real_hashtags, user_history):
    image_data_placeholder = tf.placeholder(dtype=tf.string)
    image_decoded = tf.image.decode_jpeg(image_data_placeholder, channels=3)  # channels = 3 means RGB
    image_cropped = tf.image.resize_images(image_decoded, [FLAGS.image_height, FLAGS.image_width])

    image_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.image_width, FLAGS.image_height, 3])
    image_reshaped = tf.reshape(image_placeholder, [1, FLAGS.image_width, FLAGS.image_height, 3])

    user_history_tensor = tf.reshape(tf.cast(tf.convert_to_tensor(user_history), tf.float32), [1, FLAGS.label_set_size])

    logits = createmodel.logits(image_reshaped, user_history_tensor, print_debug=False)
    logits_sig = tf.nn.sigmoid(logits)
    predictions = _get_top_predictions(logits_sig, FLAGS.predictions_count)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint_epoch = FLAGS.eval_checkpoint_epoch if FLAGS.eval_checkpoint_epoch != -1 else FLAGS.num_epochs
        checkpoint_file = "%s-%s.ckpt" % (FLAGS.checkpoint_file, checkpoint_epoch)
        saver.restore(sess, join(FLAGS.train_checkpoint_dir, checkpoint_file))

        image_cropped_out = sess.run(image_cropped, feed_dict={image_data_placeholder: image_data})
        _ = sess.run(image_reshaped, feed_dict={image_placeholder: image_cropped_out})

        _, logits_sig_out, predictions_out = sess.run([logits, logits_sig, predictions],
                              feed_dict={image_placeholder: image_cropped_out})

        print("\n---------------   Results start ---------------------\n")
        print("Real hashtags with image:\n %s\n" % real_hashtags)
        print("%20s  Prob  isCorrect" % "Prediction")
        for i in range(FLAGS.predictions_count):
            print("%20s: %4s  %s" %
                  (hashtag_name_lookup[predictions_out[0][i]],
                   logits_sig_out[0][predictions_out[0][i]],
                   hashtag_name_lookup[predictions_out[0][i]] in real_hashtags))
        print("\n---------------   Results over ---------------------\n")

        sess.close()


def main():
    hashtag_id_lookup = createHashtagsFile.get_hashtag_label_set()
    hashtag_name_lookup = {v: k for k, v in hashtag_id_lookup.iteritems()}
    real_hashtags, user_id = _get_read_image_metadata(FLAGS.image_path)
    user_history = _read_user_history()[user_id]

    with tf.gfile.FastGFile(FLAGS.image_path, 'rb') as f:
        image_data = f.read()

    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        predict_model(image_data, hashtag_name_lookup, real_hashtags, user_history)


if __name__ == "__main__":
    main()
