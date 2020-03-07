import numpy as np
import tensorflow as tf
from model import Model
from dataset import DatasetMNIST
import config

TEST_MESS = 2000
# open npy file and save it to csv

data = np.load('Signal_Costillation.npy')
print(data)
print(data.shape)
data = np.reshape(data, (1, 104))
np.savetxt('data_one_point.csv', data, delimiter=',')

# Test save and load
"""
def test(dataset_params, encoder_params, channel_params, decoder_params, decision_params):
    model = Model(dataset_params=dataset_params, encoder_params=encoder_params, channel_params=channel_params, decoder_params=decoder_params, decision_params=decision_params)
    dataset = DatasetMNIST(val_size=10000)
    data = dataset.load_data()
    decoding_mess = tf.slice(model.X, [0, 6, 0], [-1, 1, -1])

    X_batch_input = tf.math.argmax(decoding_mess, 2)
    correct_predict = tf.equal(X_batch_input, tf.math.argmax(model.get_reconstruction(), 2))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('C:\ELEC 499\Try\ML-research\ML-research-59.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('C:\ELEC 499\Try\ML-research'))

        Accuracy_SNR_6= 0
        for X_test_batch, y_test_batch in dataset.shuffle_batch(data['X_test'], data['y_test'], 100):
            Accuracy_SNR_6 += 100 * accuracy.eval(feed_dict={model.X: X_test_batch, model.Noise: 0.406})
        Accuracy_SNR_6 = 100 * Accuracy_SNR_6 / TEST_MESS
        print("SNR = 6 test accuracy is: ", 100000 * Accuracy_SNR_6)


def test_experiment():
    params = config.create_params()
    trainer = test(**params)
    trainer.train_model()


if __name__ == "__main__":
    test_experiment()
"""