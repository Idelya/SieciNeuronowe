import random
import numpy as np
from keras.utils import np_utils

from data import getData
from helper import relu, softmax, sig, tanh, sig_vec, tanh_vec, relu_vec, leaky_relu_vec
from net import Net

import matplotlib.pyplot as plt

learnig_data = [
    np.array([1, 1]),
    np.array([0, 0]),
]

HIDDEN_LAYERS_CONFIG = [
    {
        "l": 24,
        "activationFun": sig_vec,
    },
    {
        "l": 16,
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

config_normal = {
    "name": "normal",
}
def run():
    PROB=10
    EPOK=30
    charts = []
    sum_x=[]
    sum_y=[]
    sum_acc_v_x=[]
    sum_acc_v_y=[]
    sum_acc_t_x=[]
    sum_acc_t_y=[]
    for i in range(EPOK):
        sum_x.append(0)
        sum_y.append(0)
        sum_acc_v_x.append(0)
        sum_acc_v_y.append(0)
        sum_acc_t_x.append(0)
        sum_acc_t_y.append(0)

    for i in range(PROB):
        print("=====================================")
        print(i)
        print("=====================================")
        net = Net(0.01, epok=EPOK, batch=20)#alfa
        net.configLayers(784, HIDDEN_LAYERS_CONFIG, sig_vec, softmax, 10, "Xaviera")
        net.optimalizationOfAlpha(config_momentum)

        (train_X, train_y), (test_X, test_Y) = getData()
        train_Y_binary = np_utils.to_categorical(train_y)
        net.teach_me((train_X[:20000]/255, train_Y_binary[:20000]), (test_X/255, np_utils.to_categorical(test_Y)))
        charts.append([net.x_axis, net.y_axis])
        for j in range(EPOK):
            sum_x[j] = sum_x[j] + net.x_axis[j]
            sum_y[j] = sum_y[j] + net.y_axis[j]
            sum_acc_v_x[j] = sum_acc_v_x[j] + net.x_acc_v_axis[j]
            sum_acc_v_y[j] = sum_acc_v_y[j] + net.y_acc_v_axis[j]
            sum_acc_t_x[j] = sum_acc_t_x[j] + net.x_acc_t_axis[j]
            sum_acc_t_y[j] = sum_acc_t_y[j] + net.y_acc_t_axis[j]

    for i in range(PROB):
        plt.plot(charts[i][0], charts[i][1], label="Próba " + str(i+1))


    plt.title('Poprawnośc wszystkich rozwiązań')
    plt.xlabel('Epoki')
    plt.ylabel('strata')
    ax = plt.gca()
    ax.legend()
    ax.set_ylim([0, 1])
    plt.show()

    avg_y=[]
    avg_x=[]
    for i in range(EPOK):
        avg_y.append(sum_y[i]/PROB)
        avg_x.append(sum_x[i]/PROB)
    plt.plot(avg_x, avg_y, label="Średni koszt")
    plt.title('Poprawnośc 10 rozwiązań')
    plt.xlabel('Epoki')
    plt.ylabel('średnia strata')
    ax = plt.gca()
    ax.legend()
    ax.set_ylim([0, 1])
    plt.show()


    avg_y_v=[]
    avg_x_v=[]
    avg_y_t=[]
    avg_x_t=[]
    for i in range(EPOK):
        avg_y_v.append(sum_acc_v_y[i]/PROB)
        avg_x_v.append(sum_acc_v_x[i]/PROB)
        avg_y_t.append(sum_acc_t_y[i]/PROB)
        avg_x_t.append(sum_acc_t_x[i]/PROB)
    print(avg_y_v)
    print(avg_y_t)
    plt.plot(avg_x_v, avg_y_v, label="Średnia poprawność - zbiór treningowy")
    plt.plot(avg_x_t, avg_y_t, label="Średnia poprawność - zbiór walidacyjny")
    plt.title('Poprawnośc 10 rozwiązań')
    plt.xlabel('Epoki')
    plt.ylabel('średnia strata')
    ax = plt.gca()
    ax.set_ylim([0, 1])
    ax.legend()
    plt.show()
        #net2 = Net(0.01, epok=100, batch=1)#alfa
        #X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        #y = np.array([[1, 0], [0, 1], [0, 1], [0, 1]])
        #net2.configLayers(2, HIDDEN_LAYERS_CONFIG, tanh_vec, softmax, 2)
        #data = (X, y)
        #net2.teach_me(data, data)


if __name__ == '__main__':
    run()