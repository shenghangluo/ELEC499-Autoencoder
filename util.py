import tensorflow as tf


def build_neural_net(input, n_layers, n_neurons, n_outputs, activation):
    """

    :param input:
    :param n_layers:
    :param n_neurons:
    :param n_outputs:
    :param activation:
    :return:
    """
    output = input
    for i in range(n_layers):
        output = tf.contrib.layers.legacy.fully_connected(output, num_output_units=n_neurons, activation_fn=activation)

    # apply a linear transformation to last hidden layer to get the outputs
    output = tf.contrib.layers.legacy.fully_connected(output, num_output_units=n_outputs, activation_fn=None)

    return output
