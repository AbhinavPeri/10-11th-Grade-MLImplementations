from typing import List

from Layers import *
from DatasetHandler import DataHandler
from NeuralNetwork import NeuralNetwork


class PerceptronNetwork(NeuralNetwork):

    def __init__(self, input_size, loss='MSE', lr=0.01):
        super().__init__(loss, lr)
        self.layers: List[Layer] = [Input()]
        self.input_size = input_size
        self.last_layer_size = input_size

    def add_layer(self, n, activation='none'):
        self.layers.append(Dense(self.last_layer_size, n, learning_rate=self.lr))
        self.last_layer_size = n
        if activation is not 'none':
            activation_layers = {"relu": Relu, "softmax": Softmax, "sigmoid": Sigmoid}
            self.layers.append(activation_layers[activation]())

    def forward(self, x):
        new_input = x
        for i in range(len(self.layers)):
            new_input = self.layers[i].forward(new_input)
        output = new_input
        return output

    def backward(self, y, y_hat):
        current_grad = self.loss.backward(y_hat, y)
        for i in range(len(self.layers) - 1, -1, -1):
            current_grad = self.layers[i].backward(current_grad)

    def train(self, data_x, data_y, ratio=2/3, epochs=10, batch_size=20, metric='loss'):
        self.clear_logs()
        handler = DataHandler(data_x, data_y, ratio)
        for epoch in range(epochs):
            print("Epoch", epoch)
            for x_batch, y_batch in handler.iterate_mini_batches(batch_size, shuffle=True):
                output = self.forward(x_batch)
                self.backward(y_batch, output)
            self.plot_evaluation_metrics(handler, metric=metric)
        self.evaluation_plot.show()
