import tensorflow as tf
import util


class Decoder():
    def __init__(self, n_layers, n_neurons, activation, output_size):
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._activation = activation
        self._output_size = output_size

    def get_outputs(self, inputs):
        output = util.build_neural_net(input=inputs, n_layers=self._n_layers, n_neurons=self._n_neurons,
                                       activation=self._activation, n_outputs=self._output_size)
        return output
