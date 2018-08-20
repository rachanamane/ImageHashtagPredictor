
import tensorflow as tf

def logits(image_raw):
    return tf.nn.softmax(image_raw)