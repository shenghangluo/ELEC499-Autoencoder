import tensorflow as tf
import numpy as np


def gaussian_noise_layer(input_layer, std):
    #noise = tf.keras.layers.GaussianNoise(8)(input_layer)
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

