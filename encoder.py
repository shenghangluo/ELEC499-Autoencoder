import tensorflow as tf
import util


class Encoder():
    def __init__(self, n_layers, n_neurons, activation, latent_size):
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._activation = activation
        self._latent_size = latent_size

    def get_latent_representation(self, inputs):
        output = util.build_neural_net(input=inputs, n_layers=self._n_layers, n_neurons=self._n_neurons,
                                       activation=self._activation, n_outputs=self._latent_size)
        return output
