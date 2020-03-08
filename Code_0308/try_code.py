#import tensorflow as tf
# a = [[1, 10, 26.9, 2.8, 166.32, 62.3], [2, 3, 4, 5, 6, 7]]
# d = [[1, 2, 3, 4, 8, 6], [3, 4, 5, 6, 7, 8]]
# b = tf.math.argmax(a,1)
# c = tf.keras.backend.eval(b)
#
# w = tf.math.argmax(d,1)
# u = tf.keras.backend.eval(w)
#
#
# out = tf.equal(b, w)
#
#
# print("C outcome is: ", c)
# print("D outcome is:", u)
# print("is equal:", tf.keras.backend.eval(out))

# import tensorflow as tf
# import numpy as np
# import keras.backend as K
# import RToC
# # a = tf.get_variable("a", shape=[4, 4], initializer=tf.contrib.layers.xavier_initializer())
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     print(sess.run(a))
# def my_init(shape, dtype=None, partition_info=None):
#     val = np.ones((3, 3))
#     row_index = np.arange(3)
#     colum_index = np.arange(3)
#     val[row_index, colum_index] = 1.0
#
#     return K.variable(value=val, dtype=dtype)
# #
# tensor = tf.constant([[[1.0+1.0j, 2.0+1.0j, 3.0+1.0j, 4.0+1.0j], [1.0+1.0j, 2.0+1.0j, 3.0+1.0j, 4.0+1.0j]],
#                       [[1.0+1.0j, 3.0+1.0j, 5.0+1.0j, 7.0+1.0j], [1.0+1.0j, 3.0+1.0j, 5.0+1.0j, 7.0+1.0j]]])
# print("tensor shape is: ", tensor.shape)
#
# #b = tf.math.l2_normalize(tensor, axis=2)
# #print("b is: ", b.shape)
# # # c = tf.dtypes.cast(b.get_shape()[2], tf.float32)
# # # d = tf.math.sqrt(c)
# # # e = tf.multiply(d/2, b)
# #
#
# power1 = tf.math.square(tf.math.abs(tensor))
# print("power shape is: ", power1.shape)
# #power = tf.math.square(power)
# power2 = tf.math.sqrt(tf.reduce_mean(power1, axis=2, keepdims=True))
# #power = tf.math.sqrt(power)
#
# real = tf.math.real(tensor)
# imag = tf.math.imag(tensor)
#
# real = tf.math.divide(real, power2)
# imag = tf.math.divide(imag, power2)
#
# power3 = tf.complex(real, imag)
# print("power shape becomes: ", power3.shape)


# axis = list(range(len(tensor.get_shape()) - 1))
# print("axis is: ", axis)
#mean, variance = tf.nn.moments(tensor, 2, keep_dims=True)
# minus_mean=tf.math.subtract(tensor, mean)
# print("sub mean shape is: ", minus_mean.shape)
#
# after_variance=tf.math.divide(minus_mean, tf.math.sqrt(variance))
# mean_after, variance_after = tf.nn.moments(after_variance, 2, keep_dims=True)
# tensor = tf.constant([[4.0, 5.0, 6.0],
#                        [10.0, 11.0, 12.0]])
# print("tensor shape is: ", tensor.shape)
#
# a = tf.reshape(tensor, [-1,1,3])


"""Try several Encoder"""


import tensorflow as tf
import util
import config
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

            # # CtoR
            # self._real = tf.math.real(self._encoding)
            # self._real = tf.reshape(self._real, [-1, 1, 52])
            # self._image = tf.math.imag(self._encoding)
            # self._image = tf.reshape(self._image, [-1, 1, 52])
            # self._encoding = tf.concat([self._real, self._image], 2)

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


class Run_code(object):
    def __init__(self, tensor):
        self._encoder = Encoder(inputs=tensor, n_layers=1, n_neurons=256, activation= tf.nn.relu, latent_size=8)
        self._z = self._encoder.get_latent_representation()  # shape (?, 1, 104)

    def get_para(self):
        return self._z

input_tensor=tf.constant([1, 2, 3, 5])
Run_experiment = Run_code(tensor=input_tensor)
result = Run_experiment.get_para()



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("power 1 is: ", sess.run(power1))
    print("power 2 is: ", sess.run(power2))
    print("power 3 is: ", sess.run(power3))
    #print(sess.run(variance))
    #print("mean_after is: ", sess.run(mean_after))
    #print("variance_after is: ", sess.run(variance_after))
    #print("after_variance is: ", sess.run(after_variance))

