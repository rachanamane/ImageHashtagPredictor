import numpy as np
import tensorflow as tf
import preprocess.createHashtagsFile as createHashtagsFile

from model import createmodel
from os.path import isfile

# Unused import - Required for flags - Don't remove
import shared.flags
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('image_path', '/Users/namitr/tfprograms/6256005716_2018-08-08_23-22-22.jpg',
					'Directory to store TFRecords.')
flags.DEFINE_integer('predictions_count', 5,
					'Number of predictions.')


def _get_top_predictions(logits, k):
    _, indices = tf.nn.top_k(logits, k)
    return indices


def _read_hashtag_file(hash_tag_filepath):
    hash_tags = []
    with open(hash_tag_filepath, 'r') as f:
        current_hashtags = f.readline().strip().split(",")
        for current_hashtag in current_hashtags:
            current_hashtag = current_hashtag.lower()
            if current_hashtag:
                hash_tags.append(current_hashtag)
    return sorted(list(set(hash_tags)))


def _get_real_hashtags(file_path):
    if isfile(file_path) and file_path.endswith(".jpg"):
        hash_tag_filepath = file_path[0:-4] + ".txt"
        hash_tag_filepath_multiple_images = file_path[0:-6] + ".txt"
        if isfile(hash_tag_filepath):
            return _read_hashtag_file(hash_tag_filepath)
        elif isfile(hash_tag_filepath_multiple_images):
            return _read_hashtag_file(hash_tag_filepath_multiple_images)
        else:
            return []
    else:
        raise Exception("Invalid file %s" % file_path)


def predict_model(image_data, hashtag_name_lookup, real_hashtags):
    image_data_placeholder = tf.placeholder(dtype=tf.string)
    image_decoded = tf.image.decode_jpeg(image_data_placeholder, channels=3)  # channels = 3 means RGB
    image_cropped = tf.image.resize_images(image_decoded, [FLAGS.image_height, FLAGS.image_width])

    image_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.image_width, FLAGS.image_height, 3])
    image_reshaped = tf.reshape(image_placeholder, [1, FLAGS.image_width, FLAGS.image_height, 3])

    # TODO: Fix this
    user_history = [0] * FLAGS.label_set_size

    logits = createmodel.logits(image_reshaped, user_history, print_debug=False)
    logits_sig = tf.nn.sigmoid(logits)
    predictions = _get_top_predictions(logits_sig, FLAGS.predictions_count)
    #predictions = _get_top_predictions(logits, FLAGS.predictions_count)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_checkpoint_dir))

        image_cropped_out = sess.run(image_cropped, feed_dict={image_data_placeholder: image_data})
        _ = sess.run(image_reshaped, feed_dict={image_placeholder: image_cropped_out})

        _, logits_sig_out, predictions_out = sess.run([logits, logits_sig, predictions],
                              feed_dict={image_placeholder: image_cropped_out})

        print("\n---------------   Results ---------------------\n")
        print("Real hashtags with image:\n %s\n" % real_hashtags)
        print("%20s  Prob  isCorrect" % "Prediction")
        for i in range(FLAGS.predictions_count):
            print("%20s: %4s  %s" %
                  (hashtag_name_lookup[predictions_out[0][i]],
                   logits_sig_out[0][predictions_out[0][i]],
                   hashtag_name_lookup[predictions_out[0][i]] in real_hashtags))

        sess.close()


def main():
    hashtag_id_lookup = createHashtagsFile.get_hashtag_label_set()
    hashtag_name_lookup = {v: k for k, v in hashtag_id_lookup.iteritems()}
    real_hashtags = _get_real_hashtags(FLAGS.image_path)

    with tf.gfile.FastGFile(FLAGS.image_path, 'rb') as f:
        image_data = f.read()

    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        predict_model(image_data, hashtag_name_lookup, real_hashtags)


if __name__ == "__main__":
    main()
