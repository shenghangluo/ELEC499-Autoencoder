"""
Decoders for autoencoder or variational autoencoder models
"""

import tensorflow as tf
import util
import CToR
import RToC

l1 = 77
l2 = 100

class Decoder(object):
    def __init__(self, inputs, n_layers, n_neurons, activation, output_size, name='decoder'):
        with tf.name_scope(name):
            # Slicer
            self.slice_output = RToC.get_c(inputs)
            self.slice_output = tf.slice(self.slice_output, [0, l1], [-1, l2 - l1 + 1])
            #print("output from slice_output: ", self.slice_output.shape)

            # FE
            self._fe = util.build_neural_net(input=inputs, n_layers=1, n_neurons=n_neurons,
                                            activation=activation, n_outputs=output_size)
            self._fe = tf.contrib.layers.fully_connected(self._fe, num_outputs=8,
                                                         activation_fn=None)
            self._fe = RToC.get_c(self._fe)
            #print("output from FE: ", self._fe.shape)

            # PE
            self._pe = util.build_neural_net(input=inputs, n_layers=1, n_neurons=n_neurons,
                                            activation=activation, n_outputs=output_size)
            self._pe = tf.contrib.layers.fully_connected(self._pe, num_outputs=2,
                                                        activation_fn=None)
            self._pe = RToC.get_c(self._pe)

            # Mulitiply
            self._mul = tf.multiply(self.slice_output, self._pe)
            # Concatenate
            self._concat = tf.concat([self._mul, self._fe], 1)
            self._concat = CToR.get_r(self._concat)
            #print("output from concatinate: ", self._concat.shape)
            # RX
            self._decoding = util.build_neural_net(input=inputs, n_layers=n_layers, n_neurons=n_neurons,
                                                   activation=activation, n_outputs=output_size)

            #print("self._decoding shape is: ", self._decoding.shape)
            # reshape input back to (batch_size, 1, 352)
            self._decoding = tf.reshape(self._decoding, [-1, 1, 256])

    def get_outputs(self):
        return self._decoding
