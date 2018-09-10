import tensorflow as tf


def printVars(sess):
    '''
    For debugging purposes only. Not used in real code.
    Prints all tensorflow session variables
    '''
    tvars = tf.trainable_variables()
    tvars_vals = sess.run(tvars)
    for var, val in zip(tvars, tvars_vals):
        print(var.name, val)
