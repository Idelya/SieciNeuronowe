import copy

import numpy as np

from Layer import Layer

from helper import modify_train_y

class Net:
    def __init__(self, alpha):
        self.layers = []
        self.alpha = alpha

    def configLayers(self, vector_length, config_layer, start_fun, exit_fun, class_amount):
        self.layers.append(Layer(vector_length, config_layer[0]["l"], start_fun))
        for i in range(len(config_layer)):
            outputs_length = class_amount
            activation_fun = exit_fun
            if i + 1 < len(config_layer):
                outputs_length = config_layer[i + 1]["l"]
                activation_fun = config_layer[i]["activationFun"]
            self.layers.append(Layer(config_layer[i]["l"], outputs_length, activation_fun))

    def loss(self, y, teaching_data):
        sum = 0
        for i in range(len(y)):
            sum = sum + (-1 * np.log(y)*teaching_data[i])
        return sum/len(y)

    def softmax_grad(self, y, teaching_data):
        return -1*(y-teaching_data)

    def forward(self, input_vector):
        inputs = input_vector
        for layer in self.layers:
            z = layer.count_z(inputs)
            inputs = layer.get_a(z)
        return inputs

    def sample(self, input_vector, output_vector, m):
        inputs = input_vector
        for layer in self.layers:
            z = layer.count_z(inputs)
            inputs = layer.get_a(z)
        print(self.loss(inputs, output_vector))
        cost = self.softmax_grad(inputs, output_vector)*self.layers[len(self.layers)-1].a.T
        id=0
        for layer in reversed(self.layers):
            if id!=0:
                cost = layer.get_cost(cost)
                layer.upadte_weights(self.alpha, m)

    def teach_me(self, learning_data):
        for i in range(100):
            for k in range(len(learning_data[0])):
                x = np.ndarray.flatten(learning_data[0][k])
                self.sample(x, modify_train_y(learning_data[1][k]), len(learning_data))




