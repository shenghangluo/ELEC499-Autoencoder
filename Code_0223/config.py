import tensorflow as tf


def create_params():
    n_epochs = 60
    tr_batch_size = 100
    dataset_params = {'n_features': 256,
                      'validation_size': 10000}
    encoder_params = {'n_layers': 1,
                      'n_neurons': 256,
                      'activation': tf.nn.relu,
                      'latent_size': 8}

    channel_params = {'alpha': 0.33
                      }

    decoder_params = {'n_layers': 2,
                      'n_neurons': 256,
                      'activation': tf.nn.relu,
                      'output_size': 256}

    decision_params = {'n_layers': 1,
                       'n_neurons': 256,
                       'activation': tf.nn.softmax,
                       'output_size': 256}

    optimizer_params = {'lr': 1e-4}

    return {'n_epochs': n_epochs,
            'tr_batch_size': tr_batch_size,
            'dataset_params': dataset_params,
            'encoder_params': encoder_params,
            'channel_params': channel_params,
            'decoder_params': decoder_params,
            'decision_params': decision_params,
            'optimizer_params': optimizer_params}
