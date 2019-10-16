import tensorflow as tf
from dataset import DatasetMNIST
from model import Model
import loss


class Trainer(object):
    def __init__(self, n_epochs, tr_batch_size, dataset_params, encoder_params, decoder_params, optimizer_params):
        self._n_epochs = n_epochs
        self._batch_size = tr_batch_size
        self._dataset = DatasetMNIST(val_size=10000)
        self._model = Model(dataset_params=dataset_params, encoder_params=encoder_params, decoder_params=decoder_params)
        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=optimizer_params['lr'])

    def train_model(self):
        data = self._dataset.load_data()
        loss = tf.reduce_mean(tf.square(self._model.X-self._model.get_reconstruction()))
        training_op = self._optimizer.minimize(loss)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            for epoch in range(self._n_epochs):
                for X_batch, y_batch in self._dataset.shuffle_batch(data['X_train'], data['y_train'], self._batch_size):
                    sess.run(training_op, feed_dict={self._model.X: X_batch})
                print(epoch, "Training Loss:", loss.eval(feed_dict={self._model.X: data['X_valid']}))
