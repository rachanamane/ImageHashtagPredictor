import tensorflow as tf


# Unused import - Required for flags - Don't remove
import shared.flags
flags = tf.app.flags
FLAGS = flags.FLAGS

def main():
    with tf.gfile.FastGFile('/Users/namitr/tfprograms/dataset/dogsofinstagram/1502391_2018-08-08_22-35-18_3.jpg', 'rb') as f:
        image_data = f.read()

    tf.reset_default_graph()
    with tf.Graph().as_default():
        image_data_placeholder = tf.placeholder(dtype=tf.string)
        image_decoded = tf.image.decode_jpeg(image_data_placeholder, channels=3)  # channels = 3 means RGB
        image_cropped = tf.image.resize_images(image_decoded, [FLAGS.image_height, FLAGS.image_width])
        image_casted = tf.cast(image_cropped, tf.uint8)
        image_encoded = tf.image.encode_jpeg(image_casted, format='rgb', quality=100)
        print(image_cropped.shape)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            image_encoded_out = sess.run(image_encoded, feed_dict={image_data_placeholder: image_data})
            with tf.gfile.FastGFile('/Users/namitr/tfprograms/new.jpg', 'w') as f:
                f.write(image_encoded_out)

            sess.close()


if __name__ == "__main__":
    main()
