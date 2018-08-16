import createTFRecords as createTFRecords
import os
import tensorflow as tf

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(file_path, image_buffer, hash_tags, height, width):
    return tf.train.Example(features=tf.train.Features(feature={
        'height':_int64_feature(height),
        'width':_int64_feature(width),
        'labels':_int64_feature(hash_tags),
        'image_raw':_bytes_feature(image_buffer)
    }))


def _process_single_image(file_path):
    # TODO: Try changing rb to r
    with tf.gfile.FastGFile(file_path, 'rb') as f:
        image_data = f.read()

    g = tf.Graph()
    with g.as_default():
        tf.logging.set_verbosity(tf.logging.ERROR)
        sess = tf.Session()

        img_encoded = tf.placeholder(dtype=tf.string)
        t_image = tf.image.decode_jpeg(img_encoded, channels=3) # channels = 3 means RGB

        # TODO: Preserve aspect ratio here
        resized_image = tf.image.resize_images(t_image, [299, 299])
        casted_image = tf.cast(resized_image, dtype=tf.uint8)
        encoded_image = tf.image.encode_jpeg(casted_image, format='rgb', quality=100)

        sess.run(tf.initialize_all_variables())
        img = sess.run(resized_image, feed_dict={img_encoded: image_data})

        assert len(resized_image.get_shape()) == 3
        height = int(resized_image.get_shape()[0])
        width = int(resized_image.get_shape()[1])
        assert resized_image.get_shape()[2] == 3

        img = sess.run(casted_image, feed_dict={img_encoded: image_data})

        img = sess.run(encoded_image, feed_dict={img_encoded: image_data})

        sess.close()
        return img, height, width

def _process_dataset(image_and_hashtags):
    # TODO: Create flag for this path
    output_file = os.path.join('/Users/namitr/tfprograms/dataset_tfrecords', 'TFRecord')
    writer = tf.python_io.TFRecordWriter(output_file)

    index = 0
    for file_path, hash_tags in image_and_hashtags:
        image_buffer, height, width = _process_single_image(file_path)
        example = _convert_to_example(file_path, image_buffer, hash_tags, height, width)
        writer.write(example.SerializeToString())
        index += 1
        if not index % 1000:
            print("Processed %s images" % (index))


def main():
    # TODO: Create flag for root directory
    image_and_hastags = createTFRecords.read_all_directories('/Users/namitr/tfprograms/dataset')
    _process_dataset(image_and_hastags)

if __name__ == "__main__":
    main()