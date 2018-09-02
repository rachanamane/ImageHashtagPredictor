import numpy as np
import os
import sys
import tensorflow as tf
import threading

import random
import preprocess.readImages as readImages
from shared.features import ImageHashtagFeatures

# Unused import - Required for flags - Don't remove
import shared.flags

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _create_encoded_hashtags(hashtags):
    ret = np.zeros([FLAGS.label_set_size], dtype=int)
    for hashtag in hashtags:
        ret[hashtag] = 1
    return ret.tolist()


def _convert_to_example(file_path, image_buffer, hashtags, single_user_history):
    return tf.train.Example(features=tf.train.Features(feature={
        #ImageHashtagFeatures.heightFeature: _int64_feature(height),
        #ImageHashtagFeatures.widthFeature: _int64_feature(width),
        ImageHashtagFeatures.imageRawFeature: _bytes_feature(image_buffer),
        ImageHashtagFeatures.labelsFeature: _int64_feature(hashtags),
        ImageHashtagFeatures.encodedLabelsFeature: _int64_feature(_create_encoded_hashtags(hashtags)),
        ImageHashtagFeatures.userHistory: _int64_feature(single_user_history)
    }))


def _process_single_image(file_path, sess):
    # TODO: Try changing rb to r
    with tf.gfile.FastGFile(file_path, 'rb') as f:
        image_data = f.read()
        return image_data

    '''
    # Removing this, and resizing will be done when reading the TF records (during train/evaluation)

    image_encoded = tf.placeholder(dtype=tf.string)

    image_raw = tf.image.decode_jpeg(image_encoded, channels=3)  # channels = 3 means RGB

    resized_image = tf.image.resize_images(image_raw, [FLAGS.image_height, FLAGS.image_width])
    encoded_image = tf.image.encode_jpeg(resized_image, format='rgb', quality=100)

    img = sess.run(resized_image, feed_dict={image_encoded: image_data})

    assert len(resized_image.get_shape()) == 3
    height = int(resized_image.get_shape()[0])
    width = int(resized_image.get_shape()[1])
    assert resized_image.get_shape()[2] == 3

    img = sess.run(encoded_image, feed_dict={image_encoded: image_data})

    return img, height, width
    '''


def _create_tf_record_filename(mode, shard_index, num_shards):
    return '%s-image-features-%.3d-of-%.3d.tfrecord' % (mode, shard_index, num_shards)


def _process_dataset_batch(mode, image_and_hashtags, thread_index, images_per_shard, num_shards, user_history):
    output_file = os.path.join(FLAGS.tfrecords_dir, _create_tf_record_filename(mode, thread_index, num_shards))
    writer = tf.python_io.TFRecordWriter(output_file)

    image_start_index = thread_index * images_per_shard
    image_end_index = image_start_index + images_per_shard

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    images_processed = 0
    for index in range(image_start_index, image_end_index):
        file_path, hashtags, user_id = image_and_hashtags[index]
        image_buffer = _process_single_image(file_path, sess)
        example = _convert_to_example(file_path, image_buffer, hashtags, user_history[user_id])
        writer.write(example.SerializeToString())
        images_processed += 1
        if images_processed % 20 == 0:
            print("Processed %s %s images in thread %s" % (images_processed, mode, thread_index))
            sys.stdout.flush()

    sess.close()

def _process_dataset(mode, image_and_hashtags, num_shards, user_history):
    if len(image_and_hashtags) % num_shards != 0:
        raise Exception("Number of records (%s) not divisible by shards (%s) for %s " % (len(image_and_hashtags), num_shards, mode))
    images_per_shard = len(image_and_hashtags) / num_shards

    print("Launching %s threads for %s. Each thread will process %s images" % (num_shards, mode, images_per_shard))
    sys.stdout.flush()
    coord = tf.train.Coordinator()

    threads = []
    for thread_index in range(num_shards):
        args = (mode, image_and_hashtags, thread_index, images_per_shard, num_shards, user_history)
        t = threading.Thread(target=_process_dataset_batch, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)

    image_file_names = [x[0].split("/")[-2] for x in image_and_hashtags]
    dict = {}
    for x in image_file_names:
        if x in dict:
            dict[x] += 1
        else:
            dict[x] = 1
    print("Images by foldername: %s - %s" % (mode, dict))


def main():
    image_and_hashtags, user_history = readImages.read_all_directories(FLAGS.dataset_dir)
    random.shuffle(image_and_hashtags)
    if FLAGS.training_set_size + FLAGS.eval_set_size > len(image_and_hashtags):
        raise Exception("Please reduce training or evaluation size. Total images available are %s" % (len(image_and_hashtags)))
    train_image_and_hashtags = [image_and_hashtags[i] for i in range(0, FLAGS.training_set_size)]
    eval_image_and_hashtags = [image_and_hashtags[i] for i in range(FLAGS.training_set_size, FLAGS.training_set_size + FLAGS.eval_set_size)]

    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        _process_dataset('train', train_image_and_hashtags, FLAGS.train_write_shards, user_history)
        _process_dataset('eval', eval_image_and_hashtags, FLAGS.eval_write_shards, user_history)


if __name__ == "__main__":
    main()
