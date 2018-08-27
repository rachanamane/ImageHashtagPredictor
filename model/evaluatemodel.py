import numpy as np
import tensorflow as tf

from model import readTFRecords, createmodel

# Unused import - Required for flags - Don't remove
import shared.flags
FLAGS = tf.app.flags.FLAGS


def _get_top_predictions(logits, k=1):
    _, indices = tf.nn.top_k(logits, k)
    return indices


def evaluate_model():
    image_raw, _, encoded_labels = readTFRecords.read_tf_records("eval")

    image_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height, 3])
    encoded_labels_placeholder = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, FLAGS.label_set_size])

    logits = createmodel.logits(image_placeholder)
    predictions = _get_top_predictions(logits, 10)

    saver = tf.train.Saver()

    true_positives=np.zeros(10)
    false_positives=np.zeros(10)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_checkpoint_dir))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for i in range(FLAGS.eval_set_size // FLAGS.batch_size):
            image_out, encoded_labels_out = sess.run([image_raw, encoded_labels])

            logits_out, predictions_out = sess.run([logits, predictions],
                                  feed_dict={
                                      image_placeholder: image_out,
                                      encoded_labels_placeholder: encoded_labels_out})

            #print(predictions_out)

            for i in range(FLAGS.batch_size):
                for k in range(10):
                    for j in range(k+1):
                        #print(predictions_out[i][j])
                        #print[a for a in range(299) if encoded_labels_out[i][a] == 1]
                        #print(encoded_labels_out[i][predictions_out[i][j]])
                        if encoded_labels_out[i][predictions_out[i][j]] == 1:
                            true_positives[k] += 1
                        else:
                            false_positives[k] += 1
                        #print("True: %s, False: %s" % (true_positives, false_positives))
                        #print("----------------")

        for k in range(10):
            print ("Precision for top %s labels: %s%%" % (k+1, true_positives[k] * 100.0 / (true_positives[k] + false_positives[k])))

        coord.request_stop()
        coord.join(threads)
        sess.close()


def main():
    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        evaluate_model()


if __name__ == "__main__":
    main()
