"""
Decoders for autoencoder or variational autoencoder models
"""

import tensorflow as tf
import util


class Decoder(object):
    def __init__(self, inputs, n_layers, n_neurons, activation, output_size, name='decoder'):
        with tf.name_scope(name):
            self._decoding = util.build_neural_net(input=inputs, n_layers=n_layers, n_neurons=n_neurons,
                                                   activation=activation, n_outputs=output_size)

    def get_outputs(self):
        return self._decoding
