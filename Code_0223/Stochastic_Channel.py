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

            # Convolution with Root Raised Cosine Function

            filt = loadmat('rrc.mat')['rrt']
            filt = np.array(filt)
            filt = filt.flatten()[62 - 15:62 + 16]
            #print("filt shape: ", filt.shape)
            filt = tf.convert_to_tensor(filt, dtype=tf.float32)
            filt = tf.reshape(filt, [31, 1, 1])

            self._transposed = tf.transpose(self._upsampled, [0, 2, 1])
            #print("transposed shape become to: ", self._transposed.shape)
            self._convd = tf.nn.conv1d(self._transposed, filters=filt, padding='SAME')
            self._output = tf.transpose(self._convd, [0, 2, 1])
            #print("output shape finally become to: ", self._upsampled.shape)

            # Multiply with Complex number (Rotation)
            # Not implement right now

    def get_ChannelOuput(self):
        return self._upsampled

def my_init(shape, dtype=None, partition_info=None):
    val = np.zeros((104, 476))
    row_index = np.arange(52)
    colum_index = np.arange(30, 238, 4)
    val[row_index, colum_index] = 1.0

    row_index = np.arange(52, 104)
    colum_index = np.arange(268, 476, 4)
    val[row_index, colum_index] = 1.0

    return K.variable(value=val, dtype=dtype)
