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
    top_k = 10
    image_raw, _, encoded_labels = readTFRecords.read_tf_records("eval")

    image_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height, 3])
    encoded_labels_placeholder = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, FLAGS.label_set_size])

    logits = createmodel.logits(image_placeholder)
    logits_sig = tf.nn.sigmoid(logits)
    predictions = _get_top_predictions(logits_sig, top_k)
    #predictions = _get_top_predictions(logits, top_k)

    saver = tf.train.Saver()

    true_positives=np.zeros(top_k)
    false_positives=np.zeros(top_k)

    # TODO: Maybe calculate accuracy too:
    # https://stackoverflow.com/questions/50285883/tensorflow-cross-entropy-for-multi-labels-classification

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_checkpoint_dir))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        steps = FLAGS.eval_set_size // (FLAGS.batch_size)
        print("Evaluating in %s steps" % steps)
        for eval_step in range(steps):
            image_out, encoded_labels_out = sess.run([image_raw, encoded_labels])

            _, _, predictions_out = sess.run([logits, logits_sig, predictions],
                                  feed_dict={
                                      image_placeholder: image_out,
                                      encoded_labels_placeholder: encoded_labels_out})

            #print(predictions_out)

            for i in range(FLAGS.batch_size):
                for k in range(top_k):
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

            if eval_step % 20 == 19:
                print ("Precision for top 1 labels: %s%%" % (true_positives[0] * 100.0 / (true_positives[0] + false_positives[0])))

        for k in range(top_k):
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
