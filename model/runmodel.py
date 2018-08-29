from shared.features import ImageHashtagFeatures
from shared.SingleImageObject import SingleImageObject

import numpy as np
import tensorflow as tf
import time

import model.readTFRecords as readTFRecords
import model.createmodel as createmodel

import preprocess.createHashtagsFile as createHashtagsFile

from os.path import join

# Unused import - Required for flags - Don't remove
import shared.flags
FLAGS = tf.app.flags.FLAGS

def _get_top_predictions(logits, k):
    _, indices = tf.nn.top_k(logits, k)
    return indices


def run_model(predict_image_data, hashtag_name_lookup):
    # Prediction image stuff
    predictions_count = 5
    predict_image_data_placeholder = tf.placeholder(dtype=tf.string)
    predict_image_decoded = tf.image.decode_jpeg(predict_image_data_placeholder, channels=3)  # channels = 3 means RGB
    predict_image_cropped = tf.image.resize_images(predict_image_decoded, [FLAGS.image_height, FLAGS.image_width])
    predict_image_reshaped = tf.reshape(predict_image_cropped, [1, FLAGS.image_width, FLAGS.image_height, 3])
    predict_logits = createmodel.logits(predict_image_reshaped, print_debug=False)
    predict_logits_sig = tf.nn.sigmoid(predict_logits)
    predictions = _get_top_predictions(predict_logits_sig, predictions_count)

    # Start training
    image_raw, _, encoded_labels = readTFRecords.read_tf_records("train", is_training=True)

    image_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height, 3])
    encoded_labels_placeholder = tf.placeholder(tf.uint16, shape=[FLAGS.batch_size, FLAGS.label_set_size])

    logits = createmodel.logits(image_placeholder)
    loss = createmodel.loss(logits, encoded_labels_placeholder)

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    saver = tf.train.Saver()

    start_time = time.time()
    prev_time = start_time
    with tf.Session() as sess:
        # Visualize the graph through tensorboard.
        #file_writer = tf.summary.FileWriter("./logs", sess.graph)
        sess.run(tf.global_variables_initializer())

        # saver.restore(sess, FLAGS.checkpoint_file)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)


        # Initialize predict image data
        _ = sess.run(predict_image_reshaped, feed_dict={predict_image_data_placeholder: predict_image_data})


        steps = (FLAGS.training_set_size // FLAGS.batch_size)
        print("Running %s steps" % steps)
        for i in range(steps):
            image_out, encoded_labels_out = sess.run([image_raw, encoded_labels])

            _, logits_out, loss_out = sess.run(
                [train_step, logits, loss],
                feed_dict={
                    image_placeholder: image_out,
                    encoded_labels_placeholder: encoded_labels_out})

            print("Completed %s of %s steps. Loss: %s" % (i, steps, loss_out))
            if i % 20 == 19:
                saver.save(sess, join(FLAGS.train_checkpoint_dir, FLAGS.checkpoint_file))
                cur_time = time.time()
                duration = cur_time - start_time
                duration_prev = cur_time - prev_time
                prev_time = cur_time
                estimated_completion_epoch = cur_time + ((steps - i) * duration_prev / 20)
                estimated_completion = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(estimated_completion_epoch))
                print("Completed %s seconds. %s seconds for last 20 batches" % (duration, duration_prev))
                print("Estimated completion time: %s" % (estimated_completion))

                _, predict_logits_sig_out, predictions_out = sess.run([predict_logits, predict_logits_sig, predictions],
                                                              feed_dict={predict_image_data_placeholder: predict_image_data})
                print("Predictions: %s, %s, %s, %s, %s" %
                      (hashtag_name_lookup[predictions_out[0][0]],
                       hashtag_name_lookup[predictions_out[0][1]],
                       hashtag_name_lookup[predictions_out[0][2]],
                       hashtag_name_lookup[predictions_out[0][3]],
                       hashtag_name_lookup[predictions_out[0][4]]))

                #print(logits_out)

        coord.request_stop()
        coord.join(threads)
        sess.close()


def main():
    hashtag_id_lookup = createHashtagsFile.get_hashtag_label_set()
    hashtag_name_lookup = {v: k for k, v in hashtag_id_lookup.iteritems()}
    with tf.gfile.FastGFile('/Users/namitr/tfprograms/6256005716_2018-08-08_23-22-22.jpg', 'rb') as f:
        predict_image_data = f.read()

    # TODO: Consider using Estimator:
    # https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#predict
    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        run_model(predict_image_data, hashtag_name_lookup)


if __name__ == "__main__":
    main()
