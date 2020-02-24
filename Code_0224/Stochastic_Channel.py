import tensorflow as tf
import numpy as np
import keras.backend as K
from scipy.io import loadmat

class Channel(object):
    def __init__(self, inputs, alpha, name='channel'):
        with tf.name_scope(name):
            # Upsampling
            self._upsampled = tf.contrib.layers.fully_connected(inputs, num_outputs=476, activation_fn=None,
                                                             weights_initializer=my_init, biases_initializer=None, trainable=False)

            # Slice it to real part and imaginary part and do the convolution separately
            # and concate them to get the output
            self._upreal = tf.slice(self._upsampled, [0, 0, 0], [-1, -1, 238])
            self._upimage = tf.slice(self._upsampled, [0, 0, 238], [-1, -1, 238])

            # Convolution with Root Raised Cosine Function
            filt = loadmat('rrc.mat')['rrt']
            filt = np.array(filt)
            filt = filt.flatten()[62 - 15:62 + 16]
            filt = tf.convert_to_tensor(filt, dtype=tf.float32)
            filt = tf.reshape(filt, [31, 1, 1])

            # Convolution with real part
            self._transposed_real = tf.transpose(self._upreal, [0, 2, 1])
            self._convd_real = tf.nn.conv1d(self._transposed_real, filters=filt, padding='SAME')
            self._output_real = tf.transpose(self._convd_real, [0, 2, 1])

            # Convolution with imaginary part
            self._transposed_image = tf.transpose(self._upimage, [0, 2, 1])
            self._convd_image = tf.nn.conv1d(self._transposed_image, filters=filt, padding='SAME')
            self._output_image = tf.transpose(self._convd_image, [0, 2, 1])

            # Concate the real and imaginary part
            self._output = tf.concat([self._output_real, self._output_image], 2)

            # Multiply with Complex number (Rotation)
            # Not implement right now

    def get_ChannelOuput(self):
        return self._output

def my_init(shape, dtype=None, partition_info=None):
    val = np.zeros((104, 476))
    row_index = np.arange(52)
    colum_index = np.arange(30, 238, 4)
    val[row_index, colum_index] = 1.0

    row_index = np.arange(52, 104)
    colum_index = np.arange(268, 476, 4)
    val[row_index, colum_index] = 1.0

    return K.variable(value=val, dtype=dtype)
