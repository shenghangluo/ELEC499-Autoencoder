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
            #self._w = util.build_neural_net(input=self._input, n_layers=n_layers, n_neurons=n_neurons,
            #                                activation=activation, n_outputs=latent_size)
            #self._encoding = tf.contrib.layers.fully_connected(self._w, num_outputs=latent_size, activation_fn=tf.nn.relu)  # shape (?, 13, 8)

            input_1 = tf.slice(self._input, [0, 0, 0], [-1, 1, -1])
            first_dense_1 = tf.layers.dense(inputs=input_1, units=n_neurons, activation=activation, name='encoder_dense_1')
            second_dense_1 = tf.layers.dense(inputs=first_dense_1, units=latent_size, activation=activation, name='encoder_dense_2')    # shape (?, 1, 8)

            input_2 = tf.slice(self._input, [0, 1, 0], [-1, 1, -1])
            first_dense_2 = tf.layers.dense(inputs=input_2, units=n_neurons, activation=activation,
                                            name='encoder_dense_1', reuse=True)
            second_dense_2 = tf.layers.dense(inputs=first_dense_2, units=latent_size, activation=activation,
                                             name='encoder_dense_2', reuse=True)

            input_3 = tf.slice(self._input, [0, 2, 0], [-1, 1, -1])
            first_dense_3 = tf.layers.dense(inputs=input_3, units=n_neurons, activation=activation,
                                            name='encoder_dense_1', reuse=True)
            second_dense_3 = tf.layers.dense(inputs=first_dense_3, units=latent_size, activation=activation,
                                             name='encoder_dense_2', reuse=True)

            input_4 = tf.slice(self._input, [0, 3, 0], [-1, 1, -1])
            first_dense_4 = tf.layers.dense(inputs=input_4, units=n_neurons, activation=activation,
                                            name='encoder_dense_1', reuse=True)
            second_dense_4 = tf.layers.dense(inputs=first_dense_4, units=latent_size, activation=activation,
                                             name='encoder_dense_2', reuse=True)

            input_5 = tf.slice(self._input, [0, 4, 0], [-1, 1, -1])
            first_dense_5 = tf.layers.dense(inputs=input_5, units=n_neurons, activation=activation,
                                            name='encoder_dense_1', reuse=True)
            second_dense_5 = tf.layers.dense(inputs=first_dense_5, units=latent_size, activation=activation,
                                             name='encoder_dense_2', reuse=True)

            input_6 = tf.slice(self._input, [0, 5, 0], [-1, 1, -1])
            first_dense_6 = tf.layers.dense(inputs=input_6, units=n_neurons, activation=activation,
                                            name='encoder_dense_1', reuse=True)
            second_dense_6 = tf.layers.dense(inputs=first_dense_6, units=latent_size, activation=activation,
                                             name='encoder_dense_2', reuse=True)

            input_7 = tf.slice(self._input, [0, 6, 0], [-1, 1, -1])
            first_dense_7 = tf.layers.dense(inputs=input_7, units=n_neurons, activation=activation,
                                            name='encoder_dense_1', reuse=True)
            second_dense_7 = tf.layers.dense(inputs=first_dense_7, units=latent_size, activation=activation,
                                             name='encoder_dense_2', reuse=True)

            input_8 = tf.slice(self._input, [0, 7, 0], [-1, 1, -1])
            first_dense_8 = tf.layers.dense(inputs=input_8, units=n_neurons, activation=activation,
                                            name='encoder_dense_1', reuse=True)
            second_dense_8 = tf.layers.dense(inputs=first_dense_8, units=latent_size, activation=activation,
                                             name='encoder_dense_2', reuse=True)

            input_9 = tf.slice(self._input, [0, 8, 0], [-1, 1, -1])
            first_dense_9 = tf.layers.dense(inputs=input_9, units=n_neurons, activation=activation,
                                            name='encoder_dense_1', reuse=True)
            second_dense_9 = tf.layers.dense(inputs=first_dense_9, units=latent_size, activation=activation,
                                             name='encoder_dense_2', reuse=True)

            input_10 = tf.slice(self._input, [0, 9, 0], [-1, 1, -1])
            first_dense_10 = tf.layers.dense(inputs=input_10, units=n_neurons, activation=activation,
                                            name='encoder_dense_1', reuse=True)
            second_dense_10 = tf.layers.dense(inputs=first_dense_10, units=latent_size, activation=activation,
                                             name='encoder_dense_2', reuse=True)

            input_11 = tf.slice(self._input, [0, 10, 0], [-1, 1, -1])
            first_dense_11 = tf.layers.dense(inputs=input_11, units=n_neurons, activation=activation,
                                             name='encoder_dense_1', reuse=True)
            second_dense_11 = tf.layers.dense(inputs=first_dense_11, units=latent_size, activation=activation,
                                              name='encoder_dense_2', reuse=True)

            input_12 = tf.slice(self._input, [0, 11, 0], [-1, 1, -1])
            first_dense_12 = tf.layers.dense(inputs=input_12, units=n_neurons, activation=activation,
                                             name='encoder_dense_1', reuse=True)
            second_dense_12 = tf.layers.dense(inputs=first_dense_12, units=latent_size, activation=activation,
                                              name='encoder_dense_2', reuse=True)

            input_13 = tf.slice(self._input, [0, 12, 0], [-1, 1, -1])
            first_dense_13 = tf.layers.dense(inputs=input_13, units=n_neurons, activation=activation,
                                             name='encoder_dense_1', reuse=True)
            second_dense_13 = tf.layers.dense(inputs=first_dense_13, units=latent_size, activation=activation,
                                              name='encoder_dense_2', reuse=True)
            #print("second_dense_13 shape is: ", second_dense_13.shape)

            self._encoding = tf.concat([second_dense_1, second_dense_2, second_dense_3, second_dense_4, second_dense_5,     # shape (?, 13, 8)
                                        second_dense_6, second_dense_7, second_dense_8, second_dense_9, second_dense_10,
                                        second_dense_11, second_dense_12, second_dense_13], 1)
            #print("self._encoding shape is: ", self._encoding.shape)


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


