import tensorflow as tf


def create_params():
    n_epochs = 10
    tr_batch_size = 200
    dataset_params = {'n_features': 28 * 28,
                      'validation_size': 10000}
    encoder_params = {'n_layers': 2,
                      'n_neurons': 25,
                      'activation': tf.nn.relu,
                      'latent_size': 10}
    decoder_params = {'n_layers': 2,
                      'n_neurons': 25,
                      'activation': tf.nn.relu,
                      'output_size': 28 * 28}

    optimizer_params = {'lr': 1e-2}

    return {'n_epochs': n_epochs,
            'tr_batch_size': tr_batch_size,
            'dataset_params': dataset_params,
            'encoder_params': encoder_params,
            'decoder_params': decoder_params,
            'optimizer_params': optimizer_params}
