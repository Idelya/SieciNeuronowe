import random
from data import getData
import numpy as np

from net import Net

learnig_data = [
    np.array([1, 1]),
    np.array([1.1, 0.99]),
    np.array([0.89, 0.99]),
    np.array([1.123, 0.97]),
    np.array([0, 0]),
]

HIDDEN_LAYERS_CONFIG = [
    {
        "l": 3,
        "activationFun": lambda x: x,
    }
]

def run():
    for p in learnig_data:
        net = Net(p, HIDDEN_LAYERS_CONFIG)
        net.configLayers(lambda x: x)
        print(net.forward())

    #(train_X, train_y) = getData()
    #print(train_X[0][0])


if __name__ == '__main__':
    run()
