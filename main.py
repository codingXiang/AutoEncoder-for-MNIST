from Data import *
from NeuronNetwork import Model, Layer

if __name__ == "__main__":
    d = Dataset()
    mlp = Model()
    mlp.add(Layer(units=200, input_dim=784))
    mlp.add(Layer(100))
    mlp.add(Layer(50))
    mlp.add(Layer(units=10, activation='sigmoid'))
    mlp.compile(dataset=d, epoch=5000, batch_size=500, lr=0.004, momentum=0.99, end_condition=99)
    mlp.fit()