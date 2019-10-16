"""
Encoders for autoencoder or variational autoencoder models
"""

import tensorflow as tf
import util


class Encoder(object):
    def __init__(self, inputs, n_layers, n_neurons, activation, latent_size, name='encoder'):
        with tf.name_scope(name):
            self._encoding = util.build_neural_net(input=inputs, n_layers=n_layers, n_neurons=n_neurons,
                                                   activation=activation, n_outputs=latent_size)

    def get_latent_representation(self):
        return self._encoding
