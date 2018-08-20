import os
import tensorflow as tf

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


def _convert_to_example(file_path, image_buffer, hash_tags, height, width):
    return tf.train.Example(features=tf.train.Features(feature={
        ImageHashtagFeatures.heightFeature: _int64_feature(height),
        ImageHashtagFeatures.widthFeature: _int64_feature(width),
        ImageHashtagFeatures.imageRawFeature: _bytes_feature(image_buffer),
        ImageHashtagFeatures.labelsFeature: _int64_feature(hash_tags),
    }))

def _process_single_image(file_path):
    # TODO: Try changing rb to r
    with tf.gfile.FastGFile(file_path, 'rb') as f:
        image_data = f.read()

    g = tf.Graph()
    with g.as_default():
        tf.logging.set_verbosity(tf.logging.ERROR)
        sess = tf.Session()

        image_encoded = tf.placeholder(dtype=tf.string)

        image_raw = tf.image.decode_jpeg(image_encoded, channels=3)  # channels = 3 means RGB

        # TODO: Preserve aspect ratio here
        resized_image = tf.image.resize_image_with_crop_or_pad(image_raw, FLAGS.image_width, FLAGS.image_height)
        encoded_image = tf.image.encode_jpeg(resized_image, format='rgb', quality=100)

        sess.run(tf.initialize_all_variables())
        img = sess.run(resized_image, feed_dict={image_encoded: image_data})

        assert len(resized_image.get_shape()) == 3
        height = int(resized_image.get_shape()[0])
        width = int(resized_image.get_shape()[1])
        assert resized_image.get_shape()[2] == 3

        img = sess.run(encoded_image, feed_dict={image_encoded: image_data})

        sess.close()
        return img, height, width

def create_tf_record_filename():
    return 'image-features.tfrecord'

def _process_dataset(image_and_hashtags):
    output_file = os.path.join(FLAGS.tfrecords_dir, create_tf_record_filename())
    writer = tf.python_io.TFRecordWriter(output_file)

    index = 0
    for file_path, hash_tags in image_and_hashtags:
        if len(hash_tags) == 0:
            continue
        image_buffer, height, width = _process_single_image(file_path)
        example = _convert_to_example(file_path, image_buffer, hash_tags, height, width)
        writer.write(example.SerializeToString())
        index += 1
        if not index % 20:
            print("Processed %s images" % (index))


def main():
    image_and_hastags = readImages.read_all_directories(FLAGS.dataset_dir)
    # TODO: Remove this and implement batching
    image_and_hastags = [image_and_hastags[i] for i in range(0, FLAGS.training_set_size)]
    _process_dataset(image_and_hastags)

if __name__ == "__main__":
    main()