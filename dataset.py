import tensorflow as tf


class DatasetMNIST():
    def __init__(self, val_size):
        self._val_size = val_size

    def load_data(self):
        """

        :return:
        """
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
        X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
        X_valid, X_train = X_train[:self._val_size], X_train[self._val_size:]
        y_valid, y_train = y_train[:self._val_size], y_train[self._val_size:]

        return {'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'X_valid': X_valid,
                'y_valid': y_valid}
