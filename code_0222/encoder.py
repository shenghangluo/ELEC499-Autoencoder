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
            self._w = util.build_neural_net(input=inputs, n_layers=n_layers, n_neurons=n_neurons,
                                            activation=activation, n_outputs=latent_size)

            self._encoding = tf.contrib.layers.fully_connected(self._w, num_outputs=latent_size, activation_fn=tf.nn.relu)

            #Normalization
            axis = list(range(len(self._encoding.get_shape()) - 1))
            mean, variance = tf.nn.moments(self._encoding, axis)
            self._encoding = tf.nn.batch_normalization(self._encoding, mean, variance, offset=None, scale=0.7071, variance_epsilon=1e-8)
            #print("self._encoding shape is: ", self._encoding.shape)

            # Reshape-combine 13 messages
            self._encoding = tf.reshape(self._encoding, [-1, 1, group_num*latent_size])
            #print("self._encoding shape become: ", self._encoding.shape)

    def get_latent_representation(self):
        return self._encoding

