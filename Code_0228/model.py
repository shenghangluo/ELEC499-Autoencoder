import tensorflow as tf
import dataset
import encoder as enc
import Stochastic_Channel as sc
import decoder as dec
import noise as noi
import decision as decis
import CToR
import RToC

k1 = 32
k2 = 207

class Model(object):
    def __init__(self, dataset_params, encoder_params, channel_params, decoder_params, decision_params):
        self.X = tf.compat.v1.placeholder(tf.float32, shape=(None, 13, dataset_params['n_features']), name="X")
        self.Noise = tf.compat.v1.placeholder(tf.float32, shape=(), name="Noise")

        # encoder       -----Transmitter Model
        self._encoder = enc.Encoder(inputs=self.X, **encoder_params)
        self._z = self._encoder.get_latent_representation()     # shape (?, 1, 104)

        # Stochastic Channel Model
        #print("self._z", self._z.shape)
        self._channel = sc.Channel(inputs=self._z, **channel_params)
        self._channelout = self._channel.get_ChannelOuput()

        # AWGN noise layer   -----Channel Model(Part)
        w = noi.gaussian_noise_layer(input_layer=self._channelout, std=self.Noise)
        #print("w shape is: ", w.shape)
        w = tf.reshape(w, [-1, 476])
        #print("w before complex is: ", w.shape)

        # convert to complex number and then slice
        w = RToC.get_c(w)
        #print("w after complex is: ", w.shape)

        # Slicer
        self.slice_output = tf.slice(w, [0, k1], [-1, k2 - k1 + 1])
        #print("first slice is: ", self.slice_output.shape)
        # conver it back to real number
        self.slice_output = CToR.get_r(self.slice_output)
        #print("self.slice_output shape is: ", self.slice_output.shape)

        # decoder layer -input shape is (batch_size, 352)--Real
        self._decoder = dec.Decoder(inputs=self.slice_output, **decoder_params)
        u = self._decoder.get_outputs()
        # decision      -----Receiver Model
        self._decision = decis.Decision(inputs=u, **decision_params)
        self._Xhat = self._decision.get_decision()

    def get_reconstruction(self):
        return self._Xhat

    def get_para(self):
        return self._z
