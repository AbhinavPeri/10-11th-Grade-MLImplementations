import math
import numpy as np


class DataHandler:
    def __init__(self, data_x, data_y, ratio):
        self.data_x = data_x
        self.data_y = data_y
        self.length = len(data_x)
        self.train_x, self.train_y, self.test_x, self.test_y = self.split(ratio)

    def split(self, ratio):
        bound = math.ceil(ratio * self.length)
        x1 = self.data_x[:bound]
        y1 = self.data_y[:bound]
        x2 = self.data_x[bound:]
        y2 = self.data_y[bound:]
        return x1, y1, x2, y2

    def get_data_groups(self):
        return self.train_x, self.train_y, self.test_x, self.test_y

    def iterate_mini_batches(self, batch_size, shuffle=False):
        assert len(self.train_x) == len(self.train_y)
        indices = np.random.permutation(len(self.train_x))
        for start_idx in range(0, len(self.train_x) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield self.train_x[excerpt], self.train_y[excerpt]
