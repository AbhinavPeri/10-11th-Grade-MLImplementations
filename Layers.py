import numpy as np


class Layer:
    def __init__(self):
        # self.next = None
        self.input = None
        pass

    def forward(self, _input):
        self.input = _input
        return _input

    def backward(self, grad):
        return grad


class Input(Layer):
    pass


class Relu(Layer):
    def forward(self, _input):
        super().forward(_input)
        return np.maximum(0, _input)

    def backward(self, grad):
        return np.mean(grad * (self.input > 0), axis=0)


class Sigmoid(Layer):
    @classmethod
    def evaluate(cls, _input):
        return 1 / (1 + np.exp(-1 * _input))

    def forward(self, _input):
        super().forward(_input)
        return Sigmoid.evaluate(_input)

    def backward(self, grad):
        return np.mean(grad * Sigmoid.evaluate(self.input) * Sigmoid.evaluate(1 - self.input), axis=0)


class Softmax(Layer):
    def forward(self, _input):
        super().forward(_input)
        exp = np.exp(_input - np.max(_input))
        return exp / np.sum(exp, axis=1)[..., None]


class Dense(Layer):
    def __init__(self, input_size, output_size, learning_rate=0.01):
        super().__init__()
        self.lr = learning_rate
        self.w = np.random.normal(loc=0.0,
                                  scale=np.sqrt(2 / (input_size + output_size)),
                                  size=(input_size, output_size))
        self.b = np.zeros(output_size)

    def forward(self, _input):
        super().forward(_input)
        return np.dot(_input, self.w) + self.b

    def backward(self, grad):
        dx = np.dot(self.w, grad)
        dw = np.mean(np.dot(self.input[..., np.newaxis], grad[np.newaxis]), axis=0)
        db = grad
        self.w -= dw * self.lr
        self.b -= db * self.lr
        return dx


class GraphConvolution(Layer):
    def __init__(self, a_hat, input_size, output_size, learning_rate=0.01):
        super().__init__()
        self.lr = learning_rate
        self.w = Dense(input_size, output_size, learning_rate)
        self.a_hat = a_hat

    def forward(self, _input):
        super().forward(_input)
        x = self.a_hat.dot(_input)
        return self.w.forward(x)

    def backward(self, grad):
        self.w.backward(grad)
        return self.a_hat.dot(grad.dot(self.w.w.T))
