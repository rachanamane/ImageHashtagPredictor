from shared.features import ImageHashtagFeatures
from shared.SingleImageObject import SingleImageObject

import numpy as np
import tensorflow as tf
import time

import model.readTFRecords as readTFRecords
import model.createmodel as createmodel

from os.path import join

# Unused import - Required for flags - Don't remove
import shared.flags
FLAGS = tf.app.flags.FLAGS


def run_model():
    image_raw, _, encoded_labels = readTFRecords.read_tf_records("train")

    image_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height, 3])
    encoded_labels_placeholder = tf.placeholder(tf.uint16, shape=[FLAGS.batch_size, FLAGS.label_set_size])

    logits = createmodel.logits(image_placeholder)
    loss = createmodel.loss(logits, encoded_labels_placeholder)

    train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)

    saver = tf.train.Saver()

    start_time = time.time()
    with tf.Session() as sess:
        # Visualize the graph through tensorboard.
        #file_writer = tf.summary.FileWriter("./logs", sess.graph)
        sess.run(tf.global_variables_initializer())

        # saver.restore(sess, FLAGS.checkpoint_file)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        steps = (FLAGS.training_set_size * 1) // FLAGS.batch_size
        print("Running %s steps" % steps)
        # TODO: Update 1 to something appropriate
        for i in range(steps):
            image_out, encoded_labels_out = sess.run([image_raw, encoded_labels])

            _, _, loss_out = sess.run(
                [train_step, logits, loss],
                feed_dict={
                    image_placeholder: image_out,
                    encoded_labels_placeholder: encoded_labels_out})

            print("Completed %s of %s steps. Loss: %s" % (i, steps, loss_out))
            if i % 10 == 9:
                saver.save(sess, join(FLAGS.train_checkpoint_dir, FLAGS.checkpoint_file))
                duration = time.time() - start_time
                print("Completed %s seconds" % duration)

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
