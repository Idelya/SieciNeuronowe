import random
import numpy as np

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
    for p in learnig_data:
        net = Net()
        net.configLayers(2, HIDDEN_LAYERS_CONFIG, relu_vec, softmax)
        print(net.forward(p))

    #(train_X, train_y) = getData()
    #print(train_X[0][0])


if __name__ == '__main__':
    run()