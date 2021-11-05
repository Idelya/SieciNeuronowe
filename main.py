import random
import numpy as np
from keras.utils import np_utils

from data import getData
from helper import relu, softmax, sig, tanh, sig_vec, tanh_vec, relu_vec
from net import Net

learnig_data = [
    np.array([1, 1]),
    np.array([0, 0]),
]

HIDDEN_LAYERS_CONFIG = [
    {
        "l": 3,
        "activationFun": sig_vec,
    },
    {
        "l": 4,
        "activationFun": tanh_vec,
    },
]

def run():
    net = Net(0.1, epok=20)#alfa
    net.configLayers(784, HIDDEN_LAYERS_CONFIG, relu_vec, softmax, 10)

    (train_X, train_y), (test_X, test_Y) = getData()
    train_Y_binary = np_utils.to_categorical(train_y)
    net.teach_me((train_X, train_Y_binary))

    net.test_mlp(test_X[0:10], np_utils.to_categorical(test_Y[0:10]))



if __name__ == '__main__':
    run()