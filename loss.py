import tensorflow as tf


class Loss():
    def __init__(self, loss_type):
        self._loss_type = loss_type

    def _mse_loss(self, target, predictions):
        return tf.reduce_mean(tf.square(target - predictions))

    def _mae_loss(self, target, predictions):
        return tf.reduce_mean(tf.abs(target - predictions))

    def get_loss(self, target, prediction):
        if self._loss_type == 'mse':
            loss = self._mse_loss(target=target, predictions=prediction)
        elif self._loss_type == 'mae':
            loss = self._mae_loss(target=target, predictions=prediction)
        else:
            print('Unrecognized loss')

        return loss