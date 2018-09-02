import tensorflow as tf
from shared.features import ImageHashtagFeatures

# Unused import - Required for flags - Don't remove
import shared.flags
FLAGS = tf.app.flags.FLAGS


class SingleImageObject:

    def __init__(self, features_dict):
        #self.width = features_dict[ImageHashtagFeatures.widthFeature]
        #self.height = features_dict[ImageHashtagFeatures.heightFeature]
        image_encoded = features_dict[ImageHashtagFeatures.imageRawFeature]
        image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [FLAGS.image_height, FLAGS.image_width])
        #image_decoded.set_shape([FLAGS.image_width, FLAGS.image_height, 3])
        self.image_raw = image_resized
        self.labels = features_dict[ImageHashtagFeatures.labelsFeature]
        self.encoded_labels = features_dict[ImageHashtagFeatures.encodedLabelsFeature]
        self.user_history = features_dict[ImageHashtagFeatures.userHistory]