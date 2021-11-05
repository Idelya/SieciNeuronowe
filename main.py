import random
import numpy as np

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
    net = Net(0.1)
    net.configLayers(784, HIDDEN_LAYERS_CONFIG, relu_vec, softmax, 10)

    (train_X, train_y) = getData()
    net.teach_me((train_X, train_y))



if __name__ == '__main__':
    run()