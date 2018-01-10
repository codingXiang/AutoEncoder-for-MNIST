import numpy as np
from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding
from keras.datasets import mnist

class Dataset(object):
    def __init__(self):
        np.random.seed(10)
        (X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()
        self.train_x = self.input_pre_process(X_train_image)
        self.train_Y = self.one_hot_encoding(y_train_label)
        self.test_x = self.input_pre_process(X_test_image)
        self.test_Y = self.one_hot_encoding(y_test_label)
    def one_hot_encoding(self , Y):
        result = np_utils.to_categorical(Y)
        return result
    def input_pre_process(self, X):
        X = X.reshape(X.shape[0] , 784).astype('float32') / 255
        return np.array(X)