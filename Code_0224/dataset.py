import tensorflow as tf
import numpy as np
import random


class DatasetMNIST(object):
    def __init__(self, val_size):
        self._val_size = val_size

    def load_data(self):
        """
        :return:
        """
        TRAIN_MESS = 380000  # around 8,0000
        VALID_MESS = 10000
        TEST_MESS = 50000
        group_num = 13

        y_train = (np.arange(1, 257, 1))
        X_train = np.zeros((y_train.size, y_train.max()))
        X_train[np.arange(y_train.size), y_train - 1] = 1

        # print("X is ", X_train)
        # print("y is ", y_train)
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.int32)

        X_train = X_train.reshape([256, -1, 256])
        #print("X_train_shape is ", X_train.shape)
        #print("X_train is ", X_train[1,0,:])

        # X_train_all = np.zeros((TRAIN_MESS, 1, 256))
        # X_valid_all = np.zeros((VALID_MESS, 1, 256))
        # X_test_all = np.zeros((TEST_MESS, 1, 256))

        X_cons = np.zeros((20, 13, 256))
        k=0
        for i in range(20):
            for j in range(group_num):
                if k > 255:
                    X_cons[i, j, :] = 0
                else:
                    X_cons[i, j, :] = X_train[k, :, :]
                k += 1

        X_train_final = np.zeros((TRAIN_MESS, 13, 256))
        for i in range(TRAIN_MESS):
            for j in range(group_num):
                X_train_final[i, j, :] = X_train[random.randint(0, 255), :, :]

        X_valid_final = np.zeros((VALID_MESS, 13, 256))
        for i in range(VALID_MESS):
            for j in range(group_num):
                X_valid_final[i, j, :] = X_train[random.randint(0, 255), :, :]

        X_test_final = np.zeros((TEST_MESS, 13, 256))
        for i in range(TEST_MESS):
            for j in range(group_num):
                X_test_final[i, j, :] = X_train[random.randint(0, 255), :, :]

        # for i in range(TEST_MESS):
        #     X_test_all[i, :, :] = X_train[random.randint(0, 255), :, :]
        #
        # X_test_final = np.zeros((TEST_MESS - 12, 13, 256))
        # for i in range(TEST_MESS - 12):
        #     k = i
        #     for j in range(group_num):
        #         X_test_final[i, j, :] = X_test_all[k, :, :]
        #         k = k + 1
        #print("X_train_final_1 is ", X_train_final.shape)

        y_train_final = X_train_final
        y_test = X_test_final
        np.random.shuffle(y_test)
        y_valid = X_valid_final
        np.random.shuffle(y_valid)

        """
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
        X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
        X_valid, X_train = X_train[:self._val_size], X_train[self._val_size:]
        y_valid, y_train = y_train[:self._val_size], y_train[self._val_size:]
        """
        return {'X_train': X_train_final,
                'y_train': y_train_final,
                'X_test': X_test_final,
                'y_test': y_test,
                'X_valid': X_valid_final,
                'y_valid': y_valid,
                'X_cons': X_cons}

    @staticmethod
    def shuffle_batch(X, y, batch_size):
        #print("x shape is: ", X.shape)
        #print("length of x is: ", len(X))
        rnd_idx = np.random.permutation(len(X))
        n_batches = len(X) // batch_size
        #print("n_batches", n_batches)
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            yield X_batch, y_batch
