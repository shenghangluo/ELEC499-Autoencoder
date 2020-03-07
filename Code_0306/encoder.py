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
            self._encoding = tf.contrib.layers.fully_connected(self._w, num_outputs=latent_size, activation_fn=tf.nn.relu)  # shape (?, 13, 8)

            # RtoC
            self._real = tf.slice(self._encoding, [0, 0, 0], [-1, -1, 4])
            self._image = tf.slice(self._encoding, [0, 0, 4], [-1, -1, 4])
            self._complex = tf.complex(self._real, self._image)
            #print("output is: ", self._complex.shape)
            #self._encoding = self._complex
            # Normalization
            power1 = tf.math.square(tf.math.abs(self._complex))
            print("power shape is: ", power1.shape)
            # power = tf.math.square(power)
            power2 = tf.math.sqrt(tf.reduce_mean(power1, axis=2, keepdims=True))
            # power = tf.math.sqrt(power)

            real = tf.math.real(self._complex)
            imag = tf.math.imag(self._complex)
            power2 = power2 + 1e-8
            real = tf.math.divide(real, power2)
            imag = tf.math.divide(imag, power2)

            self._encoding = tf.complex(real, imag)

            # mean, variance = tf.nn.moments(self._complex, 2, keep_dims=True)
            # variance = variance+1e-8
            # minus_mean = tf.math.subtract(self._complex, mean)
            # self._encoding = tf.math.divide(minus_mean, tf.math.sqrt(variance))
            # #print("self._encoding shape is: ", self._encoding.shape)            # shape of the output from normalization is (?, 13, 4)

            # CtoR
            self._real = tf.math.real(self._encoding)
            self._real = tf.reshape(self._real, [-1, 1, 52])
            self._image = tf.math.imag(self._encoding)
            self._image = tf.reshape(self._image, [-1, 1, 52])
            self._encoding = tf.concat([self._real, self._image], 2)

            # self.normalized = tf.math.l2_normalize(self._encoding, axis=2)
            # lenth = tf.dtypes.cast(self.normalized.get_shape()[2], tf.float32)
            # sqrt_len = tf.math.sqrt(lenth)
            # self._encoding = tf.multiply(sqrt_len/2, self.normalized)

            # axis = list(range(len(self._encoding.get_shape()) - 1))
            # #mean, variance = tf.nn.moments(self._encoding, axis)
            # print("axis is: ", axis)
            # self._encoding = tf.compat.v1.layers.batch_normalization(self._encoding, axis=2, center=False, scale=False, trainable=False)
            #tf.nn.batch_normalization, tf.contrib.layers.batch_norm

            # Reshape-combine 13 messages
            #self._encoding = tf.concat([self._real, self._image], 2)
            print("self._encoding shape is: ", self._encoding.shape)            # shape of the output is (?, 1, 104)

    def get_latent_representation(self):
        return self._encoding


