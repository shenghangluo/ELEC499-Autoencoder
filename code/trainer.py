import tensorflow as tf
from dataset import DatasetMNIST
from model import Model
import numpy as np

TRAIN_MESS = 50000
VALID_MESS = 100
TEST_MESS = 1000
group_num = 13

class Trainer(object):
    def __init__(self, n_epochs, tr_batch_size, dataset_params, encoder_params, channel_params, decoder_params, decision_params, optimizer_params):
        self._n_epochs = n_epochs
        self._batch_size = tr_batch_size
        self._dataset = DatasetMNIST(val_size=10000)
        self._model = Model(dataset_params=dataset_params, encoder_params=encoder_params, channel_params=channel_params, decoder_params=decoder_params, decision_params=decision_params)
        self._optimizer = tf.train.AdamOptimizer(learning_rate=optimizer_params['lr'])      ## Tried GradientDescentOptimizer, but it seems hard to converge, but the paper mentioned SGD with Adam optimizer, so I thing the way is SGD and the optimizer is Adam
        self._writer = tf.summary.FileWriter('./summary')

    def train_model(self):
        data = self._dataset.load_data()
        # loss function
        #print("self._model.get_reconstruction()", self._model.get_reconstruction().shape)
        self._decoding_mess = tf.slice(self._model.X, [0, 6, 0], [-1, 1, -1])
        #print("self._decoding_mess shape is: ", self._decoding_mess.shape)
        loss = tf.reduce_mean(-tf.reduce_sum(self._decoding_mess * tf.log(self._model.get_reconstruction()+1e-8), 2))
        #loss = tf.reduce_mean(tf.keras.backend.categorical_crossentropy(self._model.X, self._model.get_reconstruction()))

        X_batch_input = tf.math.argmax(self._decoding_mess, 2)
        correct_predict = tf.equal(X_batch_input, tf.math.argmax(self._model.get_reconstruction(), 2))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

        tf.summary.scalar('Loss', loss)
        training_op = self._optimizer.minimize(loss)

        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            print("Training Start")
            X_train_batch = np.zeros((TRAIN_MESS, group_num, 256))
            for epoch in range(self._n_epochs):
                train_loss = 0
                if epoch <= 20:
                    for X_batch, y_batch in self._dataset.shuffle_batch(data['X_train'], data['y_train'], self._batch_size):
                        for i in range(self._batch_size-12):
                            k = i
                            for j in range(group_num):
                                X_train_batch[i, j, :] = X_batch[k, :, :]
                                k = k + 1

                        _, loss_batch = sess.run([training_op, loss], feed_dict={self._model.X: X_train_batch, self._model.Noise: 0.366})
                        train_loss += loss_batch

                    summary = sess.run(summary_op, feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.366})
                    self._writer.add_summary(summary=summary, global_step=epoch)
                    # TODO(ali): Clean this code for the LOVE PF GOD!!!
                    print(epoch, "Training Loss:", train_loss, "Validation Loss",
                          loss.eval(feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.366}), "Accuracy",
                                               100*accuracy.eval(feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.366}))

                elif epoch <= 40:
                    for X_batch, y_batch in self._dataset.shuffle_batch(data['X_train'], data['y_train'], 100):
                        for i in range(100-12):
                            k = i
                            for j in range(group_num):
                                X_train_batch[i, j, :] = X_batch[k, :, :]
                                k = k + 1

                        _, loss_batch = sess.run([training_op, loss],
                                                 feed_dict={self._model.X: X_train_batch, self._model.Noise: 0.366})
                        train_loss += loss_batch

                    summary = sess.run(summary_op, feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.366})
                    self._writer.add_summary(summary=summary, global_step=epoch)
                    # TODO(ali): Clean this code for the LOVE PF GOD!!!
                    print(epoch, "Training Loss:", train_loss, "Validation Loss",
                          loss.eval(feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.366}), "Accuracy",
                          100 * accuracy.eval(feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.366}))

                else:
                    for X_batch, y_batch in self._dataset.shuffle_batch(data['X_train'], data['y_train'], 200):
                        for i in range(200-12):
                            k = i
                            for j in range(group_num):
                                X_train_batch[i, j, :] = X_batch[k, :, :]
                                k = k + 1

                        _, loss_batch = sess.run([training_op, loss],
                                                 feed_dict={self._model.X: X_train_batch, self._model.Noise: 0.366})
                        train_loss += loss_batch

                    summary = sess.run(summary_op, feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.366})
                    self._writer.add_summary(summary=summary, global_step=epoch)
                    # TODO(ali): Clean this code for the LOVE PF GOD!!!
                    print(epoch, "Training Loss:", train_loss, "Validation Loss",
                          loss.eval(feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.366}), "Accuracy",
                          100 * accuracy.eval(feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.366}))

            # Testing
            Accuracy_SNR_0 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_0 += 100*accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.5})
            Accuracy_SNR_0 = 100*Accuracy_SNR_0/TEST_MESS
            print("SNR = 0 test accuracy is: ", 100000*Accuracy_SNR_0)

            Accuracy_SNR_1 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_1 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.483})
            Accuracy_SNR_1 = 100 * Accuracy_SNR_1 / TEST_MESS
            print("SNR = 1 test accuracy is: ", 100000 * Accuracy_SNR_1)

            Accuracy_SNR_2 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_2 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.4665})
            Accuracy_SNR_2 = 100 * Accuracy_SNR_2 / TEST_MESS
            print("SNR = 2 test accuracy is: ", 100000 * Accuracy_SNR_2)

            Accuracy_SNR_3 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_3 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.45})
            Accuracy_SNR_3 = 100 * Accuracy_SNR_3 / TEST_MESS
            print("SNR = 3 test accuracy is: ", 100000 * Accuracy_SNR_3)

            Accuracy_SNR_4 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_4 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.435})
            Accuracy_SNR_4 = 100 * Accuracy_SNR_4 / TEST_MESS
            print("SNR = 4 test accuracy is: ", 100000 * Accuracy_SNR_4)

            Accuracy_SNR_5 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_5 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.42})
            Accuracy_SNR_5 = 100 * Accuracy_SNR_5 / TEST_MESS
            print("SNR = 5 test accuracy is: ", 100000 * Accuracy_SNR_5)

            Accuracy_SNR_6 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_6 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.406})
            Accuracy_SNR_6 = 100 * Accuracy_SNR_6 / TEST_MESS
            print("SNR = 6 test accuracy is: ", 100000 * Accuracy_SNR_6)

            Accuracy_SNR_7 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_7 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.392})
            Accuracy_SNR_7 = 100 * Accuracy_SNR_7 / TEST_MESS
            print("SNR = 7 test accuracy is: ", 100000 * Accuracy_SNR_7)

            Accuracy_SNR_8 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_8 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.3789})
            Accuracy_SNR_8 = 100 * Accuracy_SNR_8 / TEST_MESS
            print("SNR = 8 test accuracy is: ", 100000 * Accuracy_SNR_8)

            Accuracy_SNR_9 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_9 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.366})
            Accuracy_SNR_9 = 100 * Accuracy_SNR_9 / TEST_MESS
            print("SNR = 9 test accuracy is: ", 100000 * Accuracy_SNR_9)

            Accuracy_SNR_10 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_10 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.353})
            Accuracy_SNR_10 = 100 * Accuracy_SNR_10 / TEST_MESS
            print("SNR = 10 test accuracy is: ", 100000 * Accuracy_SNR_10)

            Accuracy_SNR_11 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_11 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.3415})
            Accuracy_SNR_11 = 100 * Accuracy_SNR_11 / TEST_MESS
            print("SNR = 11 test accuracy is: ", 100000 * Accuracy_SNR_11)

            Accuracy_SNR_12 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_12 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.3299})
            Accuracy_SNR_12 = 100 * Accuracy_SNR_12 / TEST_MESS
            print("SNR = 12 test accuracy is: ", 100000 * Accuracy_SNR_12)

            Accuracy_SNR_13 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_13 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.3186})
            Accuracy_SNR_13 = 100 * Accuracy_SNR_13 / TEST_MESS
            print("SNR = 13 test accuracy is: ", 100000 * Accuracy_SNR_13)

            #Constellation
            for X_cons_batch, y_cons in self._dataset.shuffle_batch(data['X_cons'], data['X_cons'], 20):
                signal_cons = self._model.get_para().eval(feed_dict={self._model.X: X_cons_batch, self._model.Noise: 0.5})
            np.save('Signal_Costillation', signal_cons)
            #print("signal_cons shape is: ", signal_cons.shape)
            #print("signal_cons is: ", signal_cons)

            #accuracy.eval()