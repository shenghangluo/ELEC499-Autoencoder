import tensorflow as tf
import dataset
import encoder as enc
import decoder as dec
import loss


class Model(object):
    def __init__(self, dataset_params, encoder_params, decoder_params):
        self.X = tf.placeholder(tf.float32, shape=(None, dataset_params['n_features']), name="X")

        self._encoder = enc.Encoder(inputs=self.X, **encoder_params)
        z = self._encoder.get_latent_representation()

        self._decoder = dec.Decoder(inputs=z, **decoder_params)
        self._Xhat = self._decoder.get_outputs()

    def get_reconstruction(self):
        return self._Xhat
