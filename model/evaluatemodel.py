import numpy as np
import tensorflow as tf

import model.util as util
from model import readTFRecords, createmodel
from os.path import join

# Unused import - Required for flags - Don't remove
import shared.flags
FLAGS = tf.app.flags.FLAGS

pet_indices = [0, 1, 11, 15, 16]

only_dog_indices = [3, 9, 25, 27, 31]
all_dog_indices = only_dog_indices[:]
all_dog_indices.extend(pet_indices)

only_cat_indices = [13, 17, 18, 22, 24]
all_cat_indices = only_cat_indices[:]
all_cat_indices.extend(pet_indices)

generic_food_indices = [2, 5, 10, 26, 28]

all_food_indices = [4, 6, 7, 8, 12, 14, 19, 20, 21, 23, 29, 30]
all_food_indices.extend(generic_food_indices)

equivalent_indices = [
    # If current prediction in tuple[0], then any of tuple[1] labels should be counted as true equivalent positive.
    (only_dog_indices, all_dog_indices),
    (only_cat_indices, all_cat_indices),
    (pet_indices, pet_indices),
    (all_food_indices, generic_food_indices)
]

def _get_top_predictions(logits, k=1):
    _, indices = tf.nn.top_k(logits, k)
    return indices


def evaluate_model(cur_epoch):
    top_k = 10
    image_raw, _, encoded_labels, user_history = readTFRecords.read_tf_records("eval", is_training=False)

    image_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height, 3])
    encoded_labels_placeholder = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, FLAGS.label_set_size])
    user_history_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.label_set_size])

    logits = createmodel.logits(image_placeholder, user_history_placeholder)
    logits_sig = tf.nn.sigmoid(logits)
    predictions = _get_top_predictions(logits_sig, top_k)

    saver = tf.train.Saver()

    true_positives=np.zeros(top_k)
    equivalent_positives=np.zeros(top_k)
    false_positives=np.zeros(top_k)
    precision_denominator=np.zeros(top_k)

    # TODO: Maybe calculate accuracy too:
    # https://stackoverflow.com/questions/50285883/tensorflow-cross-entropy-for-multi-labels-classification

    #histogram = np.zeros(FLAGS.label_set_size)

    recall_numerator = 0
    recall_denominator = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint_file = "%s-%s.ckpt" % (FLAGS.checkpoint_file, cur_epoch)
        saver.restore(sess, join(FLAGS.train_checkpoint_dir, checkpoint_file))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        #util.printVars(sess)

        total_eval_images = FLAGS.images_per_shard * FLAGS.model_eval_shards
        steps = total_eval_images // FLAGS.batch_size
        print("Evaluating in %s steps" % steps)
        for eval_step in range(steps):
            image_out, encoded_labels_out, user_history_out = sess.run([image_raw, encoded_labels, user_history])

            _, _, predictions_out = sess.run([logits, logits_sig, predictions],
                                  feed_dict={
                                      image_placeholder: image_out,
                                      encoded_labels_placeholder: encoded_labels_out,
                                      user_history_placeholder: user_history_out})

            #print(predictions_out)
            for i in range(FLAGS.batch_size):  # Iterating over images in this batch
                #histogram[predictions_out[i][0]] += 1
                for k in range(top_k):  # Looking at first k predictions only
                    for j in range(k+1):  # Looking at the jth prediction in current k predictions
                        curr_prediction = predictions_out[i][j]
                        precision_denominator[k] += 1
                        #print(predictions_out[i][j])
                        #print[a for a in range(299) if encoded_labels_out[i][a] == 1]
                        #print(encoded_labels_out[i][predictions_out[i][j]])
                        if encoded_labels_out[i][curr_prediction] == 1:
                            true_positives[k] += 1
                            equivalent_positives[k] += 1
                        else:
                            marked_positive=False
                            for equivalent_indices_tuple in equivalent_indices:
                                if not marked_positive and curr_prediction in equivalent_indices_tuple[0]:
                                    for equivalent_index in equivalent_indices_tuple[1]:
                                        if encoded_labels_out[i][equivalent_index] == 1:
                                            equivalent_positives[k] += 1
                                            marked_positive=True
                                            break
                            if not marked_positive:
                                false_positives[k] += 1
                        #print("True: %s, False: %s" % (true_positives, false_positives))
                        #print("----------------")

            for i in range(FLAGS.batch_size):  # Iterating over images in this batch
                actual_label_count = sum(encoded_labels_out[i])
                recall_denominator += actual_label_count
                for j in range(actual_label_count+1):
                    curr_prediction = predictions_out[i][j]
                    if encoded_labels_out[i][curr_prediction] == 1:
                        recall_numerator += 1

            #if eval_step % 20 == 19:
            #    print ("Precision for top 1 labels (actual truth vs equivalent_hashtags_truth): %.2f%%  |  %.2f%%" %
            #           ((true_positives[0] * 100.0 / precision_denominator[0]),
            #            (equivalent_positives[0] * 100.0 / precision_denominator[0])))

        #print ("Histogram of first prediction:")
        #for i in range(FLAGS.label_set_size):
        #    print("%s: %s" % (i, histogram[i]))

        print("Results for %s epochs" % cur_epoch)
        for k in range(top_k):
            print ("Precision for top %s labels (actual truth vs equivalent_hashtags_truth): %.2f%%  |  %.2f%%" %
                       (k+1,
                        (true_positives[k] * 100.0 / precision_denominator[k]),
                        (equivalent_positives[k] * 100.0 / precision_denominator[k])))

        print("\nRecall: %.2f%%" %
              ((recall_numerator * 100 / recall_denominator)))

        print("----------------------------------------------------------------------------------")

        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(cur_epoch):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        evaluate_model(cur_epoch)


if __name__ == "__main__":
    for i in range(FLAGS.num_epochs):
        main(i+1)
