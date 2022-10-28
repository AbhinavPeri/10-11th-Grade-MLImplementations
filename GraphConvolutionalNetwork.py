from scipy.linalg import sqrtm

from Layers import *
from PerceptronNetwork import PerceptronNetwork


class GraphConvolutionalNetwork(PerceptronNetwork):

    def __init__(self, input_size, adj_mat, train_nodes, loss='MSE', lr=0.01):
        super().__init__(input_size, loss, lr)
        self.train_nodes = train_nodes
        self.a = adj_mat
        self.a = adj_mat
        self.a_mod = self.a + np.eye(adj_mat.shape[0])
        self.d_mod = np.zeros_like(self.a_mod)
        np.fill_diagonal(self.d_mod, self.a_mod.sum(axis=1).flatten())
        self.d_mod_inv_root = np.linalg.inv(sqrtm(self.d_mod))
        self.a_hat = self.d_mod_inv_root.dot(self.a_mod.dot(self.d_mod_inv_root))

    def add_layer(self, n, last_layer=False, activation='sigmoid'):
        if not last_layer:
            self.layers.append(GraphConvolution(self.a_hat, self.last_layer_size, n, learning_rate=self.lr))
        else:
            self.layers.append(Dense(self.last_layer_size, n, learning_rate=self.lr))
        self.last_layer_size = n
        if activation is not 'none':
            activation_layers = {"relu": Relu, "softmax": Softmax, "sigmoid": Sigmoid}
            self.layers.append(activation_layers[activation]())

    def get_embeddings(self):
        return self.layers[-2].input

    def evaluate(self, x, y, metric='loss', only_train=True):
        if only_train:
            output = self.forward(x)[self.train_nodes]
            y = y[self.train_nodes]
            measurement = None
            if metric == 'loss':
                measurement = self.loss.calculate(output, y)
            if metric == 'accuracy':
                example_accuracy = np.argmax(output, axis=1) == np.argmax(y, axis=1)
                measurement = example_accuracy.sum() / example_accuracy.shape[0]
            return measurement
        else:
            return super().evaluate(x, y, metric)
