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
        self._writer = tf.summary.FileWriter('./summary')

    def train_model(self):
        data = self._dataset.load_data()
        loss = tf.reduce_mean(tf.square(self._model.X-self._model.get_reconstruction()))
        tf.summary.scalar('Loss', loss)
        training_op = self._optimizer.grad(loss)

        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            for epoch in range(self._n_epochs):
                train_loss = 0
                for X_batch, y_batch in self._dataset.shuffle_batch(data['X_train'], data['y_train'], self._batch_size):
                    _, loss_batch = sess.run([training_op, loss], feed_dict={self._model.X: X_batch})
                    train_loss += loss_batch

                summary = sess.run(summary_op, feed_dict={self._model.X: data['X_valid']})
                self._writer.add_summary(summary=summary, global_step=epoch)
                # TODO(ali): Clean this code for the LOVE PF GOD!!!
                print(epoch, "Training Loss:", train_loss/250, "Validation Loss",
                      loss.eval(feed_dict={self._model.X: data['X_valid']}))
