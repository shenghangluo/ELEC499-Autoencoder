"""
Encoders for autoencoder or variational autoencoder models
"""

import tensorflow as tf
import util
group_num = 13

class Encoder(object):
    def __init__(self, inputs, n_layers, n_neurons, activation, latent_size, name='encoder'):
        with tf.name_scope(name):
            self._input = inputs
            self._w = util.build_neural_net(input=self._input, n_layers=n_layers, n_neurons=n_neurons,
                                            activation=activation, n_outputs=latent_size)
            self._encoding = tf.contrib.layers.fully_connected(self._w, num_outputs=latent_size, activation_fn=tf.nn.relu)

            # Normalization
            self.normalized = tf.math.l2_normalize(self._encoding, axis=2)
            lenth = tf.dtypes.cast(self.normalized.get_shape()[2], tf.float32)
            sqrt_len = tf.math.sqrt(lenth)
            self._encoding = tf.multiply(sqrt_len, self.normalized)

            # axis = list(range(len(self._encoding.get_shape()) - 1))
            # #mean, variance = tf.nn.moments(self._encoding, axis)
            # print("axis is: ", axis)
            # self._encoding = tf.compat.v1.layers.batch_normalization(self._encoding, axis=2, center=False, scale=False, trainable=False)
            #tf.nn.batch_normalization, tf.contrib.layers.batch_norm

            # Reshape-combine 13 messages
            self._encoding = tf.reshape(self._encoding, [-1, 1, group_num*latent_size])

    def get_latent_representation(self):
        return self._encoding


