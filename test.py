import os

import numpy as np
from PIL import Image
import PIL
from tensorflow.keras.datasets import mnist
import tensorflow.keras.layers as layers
from PerceptronNetwork import PerceptronNetwork
import matplotlib.pyplot as plt
import time
import zipfile

def mnist_ann_test():
    (x_train, y_train), (X_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)[:-50000]/255
    y_one_hot = np.zeros((10000, 10))
    y_one_hot[np.arange(10000), y_train[:-50000]] = 1
    ann = PerceptronNetwork(784, loss='CCE', lr=0.1)
    ann.add_layer(100, activation='relu')
    ann.add_layer(200, activation='relu')
    ann.add_layer(10, activation='softmax')
    ann.train(x_train, y_one_hot, ratio=4/5, epochs=10, batch_size=10, metric='accuracy')
    plt.figure(figsize=[6, 6])
    for i in range(20):
        output = np.argmax(ann.forward(X_test[i].reshape(1, 784)))
        plt.title("Actual: " + str(y_test[i]) + " Predicted: " + str(output))
        plt.imshow(X_test[i].reshape([28, 28]), cmap='gray')
        plt.show()
        time.sleep(1)

if __name__ == '__main__':
    mnist_ann_test()

