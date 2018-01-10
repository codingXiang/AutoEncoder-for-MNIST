import numpy as np
from extension import Activation, AutoEncoder, SaveData
from sklearn.metrics import accuracy_score
import time
class Model(object):
    def __init__(self):
        self.layer_list = []
        self.weight_list = []
        self.pre_delta_list = []
        self.pre_delta_bias_list = []
    def __init_parameter(self):
        self.delta_list = []
        self.delta_bias_list = []
        self.output_list = []
    def __pre_training(self,data):
        print("Start Auto Encoder....")
        start_time = time.time()
        self.__init_parameter()
        self.output_list.append(data.train_x)
        for i in range(0, self.layer_list.shape[0] - 1):
            print(" - Layer {} :".format(i + 1))
            weight_name = "weight" + str(i + 1)
            layer_name = "layer" + str(i + 1)
            w, layer = 0, 0
            try:
                print("     - load npy file....")
                w = SaveData.load_npy(filename=weight_name)
                print("     - load {} success".format(weight_name))
                layer = SaveData.load_npy(filename=layer_name)
                print("     - load {} success".format(layer_name))
            except:
                print("     - {} and {} is not exist".format(layer_name, weight_name))
                print("     - start training {}".format(layer_name))
                if (i == 0):
                    self.a = AutoEncoder(x=self.output_list[i], hidden_node=self.layer_list[i].units, lr=0.001,
                                         batch_size=200, epoch=100000, momentum=0.99, is_first_layer=True)
                else:
                    self.a = AutoEncoder(x=self.output_list[i], hidden_node=self.layer_list[i].units, lr=0.001,
                                         batch_size=200, epoch=100000, momentum=0.99, is_first_layer=False)
                w, layer = self.a.fit()
                SaveData.save_npy(filename=weight_name, data=w)
                SaveData.save_npy(filename=layer_name, data=w)
            self.weight_list.append(w)
            self.output_list.append(layer)
            self.pre_delta_bias_list.append(0)
            self.pre_delta_list.append(0)
        self.weight_list.append(self.layer_list[-1].w)
        self.pre_delta_bias_list.append(0)
        self.pre_delta_list.append(0)
        end_time = time.time()
        cost_time = end_time - start_time
        print("Auto Encoder Complete")
        print("----------------------")
        print("cost time : {} sec".format(cost_time))
        print("----------------------")
    def add(self, layer):
        count = len(self.layer_list)
        if (count > 0):
            layer.build(self.layer_list[count - 1].units)
        else:
            layer.build(layer.input_dim)
        self.layer_list.append(layer)

    def setup_weight(self):
        for i in self.layer_list:
            self.weight_list.append(i.w)
            self.pre_delta_bias_list.append(0)
            self.pre_delta_list.append(0)

    def compile(self, dataset, lr=0.1, batch_size=200, epoch=10, momentum=0.9, end_condition=98.0):
        self.end_condition = end_condition
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.momentum = momentum
        self.dataset = dataset
        self.layer_list = np.array(self.layer_list)
        self.show_info()
        self.__pre_training(self.dataset)
        self.x = self.dataset.train_x
        self.Y = self.dataset.train_Y

    def forward(self, x):
        self.__init_parameter()
        self.output_list.append(x)
        for i in range(0, self.layer_list.shape[0]):
            output = self.layer_list[i].activation(_input=np.dot(self.output_list[i], self.weight_list[i][1:]) + self.weight_list[i][0])
            self.output_list.append(output)
            self.delta_list.append(0)
        self.y = self.output_list[-1]

    def predict(self, x, Y):
        self.forward(x)
        self.accuracy = 0
        for i in range(0, Y.shape[0]):
            zy = np.argmax(self.output_list[-1][i])
            ty = np.argmax(Y[i])
            if (zy == ty):
                self.accuracy = self.accuracy + 1
        self.accuracy = self.accuracy / Y.shape[0] * 100

        return self

    def backend(self, Y):
        E = (Y - self.y)
        self.mse = E.sum() / Y.shape[0]
        delta_y = E * self.layer_list[-1].derivative(self.y)
        delta_tmp = delta_y
        self.delta_list[len(self.weight_list) - 1] = delta_tmp
        for i in range(len(self.weight_list) - 1 , 1, -1):
            delta = self.layer_list[i].derivative(self.output_list[i]) * np.dot(self.delta_list[i], self.weight_list[i][1:].T)
            self.delta_list[i - 1] = delta

        for i in range(len(self.weight_list) - 1, 0, -1):
            h = np.array(self.output_list[i])
            delta_w = self.lr * h.T.dot(np.array(self.delta_list[i]))
            delta_w_bias = self.lr * np.array(self.delta_list[i]).sum()
            self.weight_list[i][1:] += delta_w + self.pre_delta_list[i] * self.momentum
            self.weight_list[i][0] += delta_w_bias + self.pre_delta_bias_list[i] * self.momentum
            self.pre_delta_list[i] = delta_w
            self.pre_delta_bias_list[i] = delta_w_bias
        return E

    def fit(self):
        print("start training")
        self.accuracy_list = []
        start = time.time()
        for _iter in range(0, self.epoch):
            for i in range(0, self.Y.shape[0], self.batch_size):
                x = self.x[i: i + self.batch_size]
                Y = self.Y[i: i + self.batch_size]
                self.forward(x)
                self.backend(Y)
            self.predict(self.x, self.Y)
            self.accuracy_list.append(self.accuracy)
            if (_iter % 5 == 0):
                print("epoch = {} , accuracy = {:.2f}%".format(_iter, self.accuracy))
            if (self.accuracy >= self.end_condition):
                print("epoch = {} , accuracy = {:.2f}%".format(_iter, self.accuracy))
                self.testing()
                break
        print("epoch = {} , accuracy = {:.2f}%".format(_iter, self.accuracy))
        end = time.time()
        print("cost time = {:.2f} sec".format(end - start))
        self.testing()
    def testing(self):
        self.predict(self.dataset.train_x, self.dataset.train_Y)
        print("train accuracy = {:.2f}%".format(self.accuracy))
        self.predict(self.dataset.test_x, self.dataset.test_Y)
        print("test accuracy = {:.2f}%".format(self.accuracy))

    def show_info(self):
        print("mlp architecture")
        print("----------------------")
        print("| layer id | node count | weight ")
        for i in range(len(self.layer_list)):
            layer = self.layer_list[i]
            if (i == 0):
                print("|    {}     |     {}     |   {}".format((i + 1), layer.input_dim, np.array(layer.w).reshape(-1).shape))
                print("|    {}     |     {}     |   {}".format((i + 2), layer.units, np.array(self.layer_list[i + 1].w).reshape(-1).shape))
            else:
                if (i < len(self.layer_list) - 1):
                    print("|    {}     |     {}     |   {}".format((i + 2), layer.units, np.array(self.layer_list[i + 1].w).reshape(-1).shape))
                else:
                    print("|    {}     |     {}     |  ".format((i + 2), layer.units))
        print("----------------------")
        print("Parameter")
        print("----------------------")
        print("1. learning rate = {}\n2. momentum = {}\n3. epoch = {}\n4. mini batch = {}".format(self.lr, self.momentum, self.epoch, self.batch_size))
        print("----------------------")
    def data_std(self, data):
        return (data - data.mean()) / data.std()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_gradient(self, x):
        return x * (1 - x)

class Layer(object):
    def __init__(self, units = 0, input_dim = 0, activation = 'sigmoid'):
        self.units = units
        self.input_dim = input_dim
        self.a = activation

    def activation(self, _input = 0 , _label = 0):
        if (self.a == 'sigmoid'):
            return Activation.sigmoid(x=_input)
        if (self.a == 'softmax'):
            return Activation.softmax(_input)
        if (self.a == 'relu'):
            return Activation.relu(_input)
    def derivative(self, x):
        if (self.a == 'sigmoid' or self.a == 'softmax'):
            return Activation.sigmoid_gradient(x)
        if (self.a == 'relu'):
            return Activation.relu_derivative(x)
    def build(self , input_dim):
        self.w = np.random.uniform(-1.0, 1.0, size=self.units * (input_dim + 1))
        self.w = self.w.reshape(input_dim + 1, self.units)

class AutoLayer(Layer):
    def __init__(self, w, layer):
        self.__w = w
        self.__layer = layer
    def get_weight(self):
        return self.__w
    def get_layer(self):
        return self.__layer