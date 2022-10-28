from matplotlib import pyplot as plt

from Loss import *


class NeuralNetwork:
    def __init__(self, loss='MSE', lr=0.01):
        loss_classes = {"MSE": MeanSquaredError, "CCE": CategoricalCrossEntropy}
        self.loss = loss_classes[loss]
        self.lr = lr
        self.train_log = []
        self.test_log = []
        plt.legend(loc='best')
        plt.grid()
        self.evaluation_plot = plt

    def forward(self, _input) -> np.ndarray:
        pass

    def backward(self, y, y_hat):
        pass

    def evaluate(self, x, y, metric='loss'):
        output = self.forward(x)
        measurement = None
        if metric == 'loss':
            measurement = self.loss.calculate(output, y)
        if metric == 'accuracy':
            example_accuracy = np.argmax(output, axis=1) == np.argmax(y, axis=1)
            measurement = example_accuracy.sum() / example_accuracy.shape[0]
        return measurement

    def clear_logs(self):
        self.train_log = []
        self.test_log = []
        pass

    def train(self, data_x, data_y, ratio=2 / 3, epochs=10, batch_size=20, metric='loss'):
        pass

    def plot_evaluation_metrics(self, data_handler, metric='loss'):
        train_x, train_y, test_x, test_y = data_handler.get_data_groups()
        train_metric = self.evaluate(train_x, train_y, metric=metric)
        test_metric = self.evaluate(test_x, test_y, metric=metric)
        self.train_log.append(train_metric)
        self.test_log.append(test_metric)
        print("Train " + metric + ":", train_metric)
        print("Val " + metric + ":", test_metric)
        self.evaluation_plot.plot(self.train_log, label='train ' + metric)
        self.evaluation_plot.plot(self.test_log, label='val ' + metric)
