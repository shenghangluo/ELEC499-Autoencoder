import tensorflow as tf
from dataset import DatasetMNIST
from model import Model
import numpy as np

TRAIN_MESS = 200000  # around 8,0000
VALID_MESS = 8000
TEST_MESS = 30000
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
        #print("self._model.get_reconstruction shape is: ", self._model.get_reconstruction().shape)
        loss = tf.reduce_mean(-1.0*tf.reduce_sum(self._decoding_mess * tf.log(self._model.get_reconstruction()+1e-8), 2))
        #loss = tf.reduce_mean(tf.keras.backend.categorical_crossentropy(self._model.X, self._model.get_reconstruction()))

        X_batch_input = tf.math.argmax(self._decoding_mess, 2)
        correct_predict = tf.equal(X_batch_input, tf.math.argmax(self._model.get_reconstruction(), 2))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

        tf.summary.scalar('Loss', loss)
        training_op = self._optimizer.minimize(loss)
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            print("Training Start")
            for epoch in range(0, self._n_epochs):
                train_loss = 0
                if epoch <= 20:
                    for X_batch, y_batch in self._dataset.shuffle_batch(data['X_train'], data['y_train'], self._batch_size):
                        _, loss_batch = sess.run([training_op, loss], feed_dict={self._model.X: X_batch, self._model.Noise: 0.1774})
                        #print("self._model.get_reconstruction() is: ", self._model.get_reconstruction().eval(feed_dict={self._model.X: X_batch, self._model.Noise: 0.1774}))
                        train_loss += loss_batch

                    summary = sess.run(summary_op, feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.1774})
                    self._writer.add_summary(summary=summary, global_step=epoch)
                    # TODO(ali): Clean this code for the LOVE PF GOD!!!
                    print(epoch, "Training Loss:", train_loss, "Validation Loss",
                          loss.eval(feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.1774}), "Accuracy",
                                               100*accuracy.eval(feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.1774}))

                elif epoch <= 40:
                    for X_batch, y_batch in self._dataset.shuffle_batch(data['X_train'], data['y_train'], 100):
                        _, loss_batch = sess.run([training_op, loss],
                                                 feed_dict={self._model.X: X_batch, self._model.Noise: 0.1774})
                        train_loss += loss_batch

                    summary = sess.run(summary_op, feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.1774})
                    self._writer.add_summary(summary=summary, global_step=epoch)
                    # TODO(ali): Clean this code for the LOVE PF GOD!!!
                    print(epoch, "Training Loss:", train_loss, "Validation Loss",
                          loss.eval(feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.1774}), "Accuracy",
                          100 * accuracy.eval(feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.1774}))

                else:
                    for X_batch, y_batch in self._dataset.shuffle_batch(data['X_train'], data['y_train'], 200):
                        _, loss_batch = sess.run([training_op, loss],
                                                 feed_dict={self._model.X: X_batch, self._model.Noise: 0.1774})
                        train_loss += loss_batch

                    summary = sess.run(summary_op, feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.1774})
                    self._writer.add_summary(summary=summary, global_step=epoch)
                    # TODO(ali): Clean this code for the LOVE PF GOD!!!
                    print(epoch, "Training Loss:", train_loss, "Validation Loss",
                          loss.eval(feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.1774}), "Accuracy",
                          100 * accuracy.eval(feed_dict={self._model.X: data['X_valid'], self._model.Noise: 0.1774}))
                saver.save(sess, "C:/ELEC 499/Try/ML-research", global_step=epoch)
            # Testing
            Accuracy_SNR_0 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_0 += 100*accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.5})
            Accuracy_SNR_0 = 100*Accuracy_SNR_0/TEST_MESS
            print("SNR = 0 test accuracy is: ", 100000*Accuracy_SNR_0)
            np.save('SNR0_Accuracy', 100000 * Accuracy_SNR_0)

            Accuracy_SNR_1 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_1 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.4456})
            Accuracy_SNR_1 = 100 * Accuracy_SNR_1 / TEST_MESS
            print("SNR = 1 test accuracy is: ", 100000 * Accuracy_SNR_1)
            np.save('SNR1_Accuracy', 100000 * Accuracy_SNR_1)

            Accuracy_SNR_2 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_2 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.397})
            Accuracy_SNR_2 = 100 * Accuracy_SNR_2 / TEST_MESS
            print("SNR = 2 test accuracy is: ", 100000 * Accuracy_SNR_2)
            np.save('SNR2_Accuracy', 100000 * Accuracy_SNR_2)

            Accuracy_SNR_3 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_3 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.354})
            Accuracy_SNR_3 = 100 * Accuracy_SNR_3 / TEST_MESS
            print("SNR = 3 test accuracy is: ", 100000 * Accuracy_SNR_3)
            np.save('SNR3_Accuracy', 100000 * Accuracy_SNR_3)

            Accuracy_SNR_4 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_4 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.315})
            Accuracy_SNR_4 = 100 * Accuracy_SNR_4 / TEST_MESS
            print("SNR = 4 test accuracy is: ", 100000 * Accuracy_SNR_4)
            np.save('SNR4_Accuracy', 100000 * Accuracy_SNR_4)

            Accuracy_SNR_5 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_5 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.281})
            Accuracy_SNR_5 = 100 * Accuracy_SNR_5 / TEST_MESS
            print("SNR = 5 test accuracy is: ", 100000 * Accuracy_SNR_5)
            np.save('SNR5_Accuracy', 100000 * Accuracy_SNR_5)

            Accuracy_SNR_6 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_6 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.2506})
            Accuracy_SNR_6 = 100 * Accuracy_SNR_6 / TEST_MESS
            print("SNR = 6 test accuracy is: ", 100000 * Accuracy_SNR_6)
            np.save('SNR6_Accuracy', 100000 * Accuracy_SNR_6)

            Accuracy_SNR_7 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_7 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.2233})
            Accuracy_SNR_7 = 100 * Accuracy_SNR_7 / TEST_MESS
            print("SNR = 7 test accuracy is: ", 100000 * Accuracy_SNR_7)
            np.save('SNR7_Accuracy', 100000 * Accuracy_SNR_7)

            Accuracy_SNR_8 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_8 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.199})
            Accuracy_SNR_8 = 100 * Accuracy_SNR_8 / TEST_MESS
            print("SNR = 8 test accuracy is: ", 100000 * Accuracy_SNR_8)
            np.save('SNR8_Accuracy', 100000 * Accuracy_SNR_8)

            Accuracy_SNR_9 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_9 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.1774})
            Accuracy_SNR_9 = 100 * Accuracy_SNR_9 / TEST_MESS
            print("SNR = 9 test accuracy is: ", 100000 * Accuracy_SNR_9)
            np.save('SNR9_Accuracy', 100000 * Accuracy_SNR_9)

            Accuracy_SNR_10 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_10 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.158})
            Accuracy_SNR_10 = 100 * Accuracy_SNR_10 / TEST_MESS
            print("SNR = 10 test accuracy is: ", 100000 * Accuracy_SNR_10)
            np.save('SNR10_Accuracy', 100000 * Accuracy_SNR_10)

            Accuracy_SNR_11 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_11 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.141})
            Accuracy_SNR_11 = 100 * Accuracy_SNR_11 / TEST_MESS
            print("SNR = 11 test accuracy is: ", 100000 * Accuracy_SNR_11)
            np.save('SNR11_Accuracy', 100000 * Accuracy_SNR_11)

            Accuracy_SNR_12 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_12 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.1256})
            Accuracy_SNR_12 = 100 * Accuracy_SNR_12 / TEST_MESS
            print("SNR = 12 test accuracy is: ", 100000 * Accuracy_SNR_12)
            np.save('SNR12_Accuracy', 100000 * Accuracy_SNR_12)
            
            Accuracy_SNR_13 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_13 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.112})
            Accuracy_SNR_13 = 100 * Accuracy_SNR_13 / TEST_MESS
            print("SNR = 13 test accuracy is: ", 100000 * Accuracy_SNR_13)
            np.save('SNR13_Accuracy', 100000 * Accuracy_SNR_13)
            
            Accuracy_SNR_14 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_14 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.1})
            Accuracy_SNR_14 = 100 * Accuracy_SNR_14 / TEST_MESS
            print("SNR = 14 test accuracy is: ", 100000 * Accuracy_SNR_14)
            np.save('SNR14_Accuracy', 100000 * Accuracy_SNR_14)
            
            Accuracy_SNR_15 = 0
            for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
                Accuracy_SNR_15 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.089})
            Accuracy_SNR_15 = 100 * Accuracy_SNR_15 / TEST_MESS
            print("SNR = 15 test accuracy is: ", 100000 * Accuracy_SNR_15)
            np.save('SNR15_Accuracy', 100000 * Accuracy_SNR_15)
            
            # Accuracy_SNR_16 = 0
            # for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
            #     Accuracy_SNR_16 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.079})
            # Accuracy_SNR_16 = 100 * Accuracy_SNR_16 / TEST_MESS
            # print("SNR = 16 test accuracy is: ", 100000 * Accuracy_SNR_16)
            # np.save('SNR16_Accuracy', 100000 * Accuracy_SNR_16)
            #
            # Accuracy_SNR_17 = 0
            # for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
            #     Accuracy_SNR_17 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.0706})
            # Accuracy_SNR_17 = 100 * Accuracy_SNR_17 / TEST_MESS
            # print("SNR = 17 test accuracy is: ", 100000 * Accuracy_SNR_17)
            # np.save('SNR17_Accuracy', 100000 * Accuracy_SNR_17)
            #
            # Accuracy_SNR_18 = 0
            # for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
            #     Accuracy_SNR_18 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.0629})
            # Accuracy_SNR_18 = 100 * Accuracy_SNR_18 / TEST_MESS
            # print("SNR = 18 test accuracy is: ", 100000 * Accuracy_SNR_18)
            # np.save('SNR18_Accuracy', 100000 * Accuracy_SNR_18)
            #
            #
            # Accuracy_SNR_19 = 0
            # for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
            #     Accuracy_SNR_19 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.0561})
            # Accuracy_SNR_19 = 100 * Accuracy_SNR_19 / TEST_MESS
            # print("SNR = 19 test accuracy is: ", 100000 * Accuracy_SNR_19)
            # np.save('SNR19_Accuracy', 100000 * Accuracy_SNR_19)
            #
            # Accuracy_SNR_20 = 0
            # for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
            #     Accuracy_SNR_20 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.05})
            # Accuracy_SNR_20 = 100 * Accuracy_SNR_20 / TEST_MESS
            # print("SNR = 20 test accuracy is: ", 100000 * Accuracy_SNR_20)
            # np.save('SNR20_Accuracy', 100000 * Accuracy_SNR_20)
            #
            # Accuracy_SNR_21 = 0
            # for X_test_batch, y_test_batch in self._dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
            #     Accuracy_SNR_21 += 100 * accuracy.eval(feed_dict={self._model.X: X_test_batch, self._model.Noise: 0.04456})
            # Accuracy_SNR_21 = 100 * Accuracy_SNR_21 / TEST_MESS
            # print("SNR = 21 test accuracy is: ", 100000 * Accuracy_SNR_21)
            # np.save('SNR21_Accuracy', 100000 * Accuracy_SNR_21)

            #Constellation
            #for X_cons_batch, y_cons in self._dataset.shuffle_batch(data['X_cons'], data['X_cons'], 20):
            signal_cons = self._model.get_para().eval(feed_dict={self._model.X: data['X_cons'], self._model.Noise: 0.5})
            np.save('Signal_Costillation', signal_cons)





