from shared.features import ImageHashtagFeatures
from shared.singleimageobject import SingleImageObject

import numpy as np
import tensorflow as tf

import model.readTFRecords as readTFRecords
import model.createmodel as createmodel
import model.flower as flower

# Unused import - Required for flags - Don't remove
import shared.flags
FLAGS = tf.app.flags.FLAGS


def run_model():
    image_raw, labels = readTFRecords.read_tf_records()
    print(image_raw.shape)
    print(labels.shape)

    image_placeholder = tf.placeholder(tf.float32, shape=[10, FLAGS.image_width, FLAGS.image_height, 3])
    labels_placeholder = tf.placeholder(tf.uint16, shape=[10, 5])

    #logits = createmodel.logits(image_placeholder)
    logits = flower.flower_inference(image_placeholder)
    loss = tf.losses.mean_squared_error(labels=labels_placeholder, predictions=logits)

    train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Visualize the graph through tensorboard.
        file_writer = tf.summary.FileWriter("./logs", sess.graph)
        sess.run(tf.global_variables_initializer())

        #saver.restore(sess, FLAGS.checkpoint_file)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)

        for i in range(FLAGS.training_set_size * 100):
            image_out, label_out = sess.run([image_raw, labels])

            print(image_out.shape)
            _, infer_out, loss_out = sess.run(
                    [train_step, logits, loss],
                    feed_dict={
                        image_placeholder: image_out,
                        labels_placeholder: label_out})

            print(i)
            print("infer_out: ")
            print(infer_out)
            print("loss: ")
            print(loss_out)
            if(i%20 == 0):
                saver.save(sess, FLAGS.checkpoint_file)

        coord.request_stop()
        coord.join(threads)
        sess.close()

def main():
    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        run_model()

if __name__ == "__main__":
    main()