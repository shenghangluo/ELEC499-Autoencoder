import tensorflow as tf
import RToC

def get_r(inputs):
    real = tf.math.real(inputs)
    imag = tf.math.imag(inputs)
    output = tf.concat([real, imag], 1)
    return output

