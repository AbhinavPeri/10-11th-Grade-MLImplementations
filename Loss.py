import numpy as np


class Loss:
    @classmethod
    def calculate(cls, predicted, actual):
        pass

    @classmethod
    def backward(cls, predicted, actual):
        pass


class CategoricalCrossEntropy(Loss):
    @classmethod
    def calculate(cls, predicted, actual):
        return -1 * np.mean(np.sum(actual * np.log(predicted), axis=1))

    @classmethod
    def backward(cls, predicted, actual):
        return np.mean(predicted - actual, axis=0)


class MeanSquaredError(Loss):
    @classmethod
    def calculate(cls, predicted, actual):
        return np.mean((actual - predicted)**2)

    @classmethod
    def backward(cls, predicted, actual):
        if not hasattr(actual, '__len__'):
            length = 1
        else:
            length = len(actual)
        return np.mean(-2/length * (actual - predicted), axis=0)
