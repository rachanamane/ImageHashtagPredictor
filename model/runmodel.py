import tensorflow as tf
import time

import model.readTFRecords as readTFRecords
import model.createmodel as createmodel
import model.util as util

from os.path import join

# Unused import - Required for flags - Don't remove
import shared.flags
FLAGS = tf.app.flags.FLAGS

def _get_top_predictions(logits, k):
    _, indices = tf.nn.top_k(logits, k)
    return indices


def run_model():
    image_raw, _, encoded_labels, user_history = readTFRecords.read_tf_records("train", is_training=True)

    image_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height, 3])
    encoded_labels_placeholder = tf.placeholder(tf.uint16, shape=[FLAGS.batch_size, FLAGS.label_set_size])
    user_history_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.label_set_size])

    logits = createmodel.parallel_conv2d_logits(image_placeholder, user_history_placeholder)
    loss = createmodel.loss(logits, encoded_labels_placeholder)

    train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)

    saver = tf.train.Saver()

    start_time = time.time()
    prev_time = start_time
    with tf.Session() as sess:
        # Visualize the graph through tensorboard.
        file_writer = tf.summary.FileWriter(FLAGS.tensorboard_logs_dir, sess.graph)
        sess.run(tf.global_variables_initializer())

        # saver.restore(sess, FLAGS.checkpoint_file)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        steps = (FLAGS.training_set_size * FLAGS.num_epochs // FLAGS.batch_size)
        print("Running %s steps" % steps)
        for i in range(steps):
            image_out, encoded_labels_out, user_history_out = sess.run([image_raw, encoded_labels, user_history])

            _, logits_out, loss_out = sess.run(
                [train_step, logits, loss],
                feed_dict={
                    image_placeholder: image_out,
                    encoded_labels_placeholder: encoded_labels_out,
                    user_history_placeholder: user_history_out})

            print("Completed %s of %s steps. Loss: %s" % (i, steps, loss_out))
            if i % 20 == 19:
                saver.save(sess, join(FLAGS.train_checkpoint_dir, FLAGS.checkpoint_file))
                #util.printVars(sess)
                cur_time = time.time()
                duration = cur_time - start_time
                duration_prev = cur_time - prev_time
                prev_time = cur_time
                estimated_completion_epoch = cur_time + ((steps - i) * duration_prev / 20)
                estimated_completion = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(estimated_completion_epoch))
                print("Completed %s seconds. %s seconds for last 20 batches" % (duration, duration_prev))
                print("Estimated completion time: %s" % (estimated_completion))

        coord.request_stop()
        coord.join(threads)
        sess.close()


def main():
    # TODO: Consider using Estimator:
    # https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#predict
    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        run_model()


if __name__ == "__main__":
    main()
