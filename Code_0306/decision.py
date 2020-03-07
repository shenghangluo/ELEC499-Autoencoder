"""
Encoders for autoencoder or variational autoencoder models
"""

import tensorflow as tf
import util


class Decision(object):
    def __init__(self, inputs, n_layers, n_neurons, activation, output_size, name='decision'):
        with tf.name_scope(name):
            self._decision = util.build_neural_net(input=inputs, n_layers=n_layers, n_neurons=n_neurons,
                                                   activation=activation, n_outputs=output_size)

    def get_decision(self):
        return self._decision



