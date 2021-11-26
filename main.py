import random
import numpy as np
from keras.utils import np_utils

from data import getData
from helper import relu, softmax, sig, tanh, sig_vec, tanh_vec, relu_vec
from net import Net

import matplotlib.pyplot as plt

learnig_data = [
    np.array([1, 1]),
    np.array([0, 0]),
]

HIDDEN_LAYERS_CONFIG = [
    {
        "l": 10,
        "activationFun": None,
    },
]

config_momentum = {
    "name": "momentum",
    "last_update": [],
    "last_update_bias": [],
    "factor": 0.7,
}

config_momentum_nesterova = {
    "name": "momentum_nesterova",
    "last_update": [],
    "last_update_bias": [],
    "factor": 0.7,
}


config_adagrad = {
    "name": "adagrad",
}

config_adadelta = {
    "name": "adadelta",
    "factor": 0.9,
}


config_adam = {
    "name": "adam",
    "beta1": 0.9,
    "beta2": 0.9,
}
def run():
    for i in range(1):
        print("=====================================")
        print(i)
        print("=====================================")
        net = Net(0.01, epok=10, batch=20)#alfa
        net.configLayers(784, HIDDEN_LAYERS_CONFIG, tanh_vec, softmax, 10)
        net.optimalizationOfAlpha(config_momentum_nesterova)

        (train_X, train_y), (test_X, test_Y) = getData()
        train_Y_binary = np_utils.to_categorical(train_y)
        net.teach_me((train_X/255, train_Y_binary), (test_X/255, np_utils.to_categorical(test_Y)))

        plt.plot(net.xAxis, net.yAxis)
        plt.title('Poprawnośc rozwiązań')
        plt.xlabel('Epoki')
        plt.ylabel('strata')
        plt.show()
        #net2 = Net(0.01, epok=100, batch=1)#alfa
        #X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        #y = np.array([[1, 0], [0, 1], [0, 1], [0, 1]])
        #net2.configLayers(2, HIDDEN_LAYERS_CONFIG, tanh_vec, softmax, 2)
        #data = (X, y)
        #net2.teach_me(data, data)


if __name__ == '__main__':
    run()