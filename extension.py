import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
import tensorflow as tf
from keras import backend as K
class SaveData(object):
    @staticmethod
    def save_npy(filename, data):
        filename = filename + ".npy"
        np.save(filename, data)

    @staticmethod
    def load_npy(filename):
        filename = filename + ".npy"
        return np.load(filename)
class Activation(object):
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_gradient(x):
        return x * (1 - x)

    @staticmethod
    def softmax(x):
        x_exp = np.exp(x)
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)

    @staticmethod
    def relu(x):
        return x * (x > 0)
    @staticmethod
    def relu_derivative(x):
        return 1. * (x > 0)

class AutoEncoder(object):
    def __init__(self, x, lr=0.01, batch_size=200, epoch=1000, momentum=0.99, hidden_node=200, is_first_layer=False):
        self.is_first_layer = is_first_layer
        if self.is_first_layer:
            self.x = self.__data_std(x)
        else:
            self.x = self.__data_std(x)
        self.h_node = hidden_node
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.momentum = momentum
        self.setup()
    def __data_std(self, x):
        x = np.where(x > 0, 1, 0)
        return x

    def setup(self):
        self.setup_network()
        self.setup_weight()

    def setup_network(self):
        self.x_node = self.x.shape[1]
        self.pre_delta_y = 0
        self.pre_delta_h = 0
        self.pre_delta_y_bias = 0
        self.pre_delta_h_bias = 0

    def setup_weight(self):
        self.w1 = np.random.uniform(-1.0, 1.0, size=self.h_node * (self.x_node + 1))
        self.w1 = self.w1.reshape(self.x_node + 1, self.h_node)
        self.w2 = np.random.uniform(-1.0, 1.0, size=self.x_node * (self.h_node + 1))
        self.w2 = self.w2.reshape((self.h_node + 1, self.x_node))

    def forward(self, x):
        self.h = self.sigmoid((np.dot(x, self.w1[1:]) + self.w1[0]))
        self.y = self.sigmoid((np.dot(self.h, self.w2[1:]) + self.w2[0]))

    def predict(self, x):
        self.forward(x)
        self.mse = mean_squared_error(x, self.y)
        return self

    def backend(self, x):
        E = (x - self.y)
        delta_y = E * self.sigmoid_gradient(self.y)
        delta_h = self.sigmoid_gradient(self.h) * np.dot(delta_y, self.w2[1:].T)

        self.w2[1:] += self.lr * self.h.T.dot(delta_y) + (self.pre_delta_y * self.momentum)
        self.w1[1:] += self.lr * x.T.dot(delta_h) + (self.pre_delta_h * self.momentum)

        self.w2[0] += self.lr * delta_y.sum() + (self.momentum * self.pre_delta_y_bias)
        self.w1[0] += self.lr * delta_h.sum() + (self.momentum * self.pre_delta_h_bias)

        self.pre_delta_y = self.lr * self.h.T.dot(delta_y)
        self.pre_delta_h = self.lr * x.T.dot(delta_h)

        self.pre_delta_y_bias = self.lr * delta_y.sum()
        self.pre_delta_h_bias = self.lr * delta_h.sum()
        return E

    def fit(self):
        for _iter in range(0, self.epoch):
            for i in range(0, self.x.shape[0], self.batch_size):
                x = self.x[i: i + self.batch_size]
                self.forward(x)
                self.backend(x)
            self.predict(self.x)

            if (self.is_first_layer):
                if (_iter % 5 == 0):
                    print("     - epoch = {} , mse = {}".format(_iter, self.mse))
                if (self.mse <= 0.01):
                    print("     - epoch = {} , mse = {}".format(_iter, self.mse))
                    return [self.w1, self.h]
            else:
                if (_iter % 1000 == 0):
                    print("     - epoch = {} , mse = {}".format(_iter, self.mse))
                if (self.mse <= 1e-7):
                    print("     - epoch = {} , mse = {}".format(_iter, self.mse))
                    return [self.w1, self.h]
        return [self.w1, self.h]

    def data_std(self, data):
        return (data - data.mean()) / data.std()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_gradient(self, x):
        return x * (1 - x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def relu(self, x, alpha=0., max_value=None):
        if alpha != 0.:
            negative_part = tf.nn.relu(-x)
        x = tf.nn.relu(x)
        if max_value is not None:
            max_value = _to_tensor(max_value, x.dtype.base_dtype)
            zero = _to_tensor(0., x.dtype.base_dtype)
            x = tf.clip_by_value(x, zero, max_value)
        if alpha != 0.:
            alpha = _to_tensor(alpha, x.dtype.base_dtype)
            x -= alpha * negative_part
        return tf.Session().run(x)

    def relu_derivative(self, x):
        return 1. * (x > 0)