import copy

import numpy as np

from Layer import Layer


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
        return -1*max(y)-max(teaching_data)

    def forward(self, input_vector):
        inputs = input_vector
        for layer in self.layers:
            z = layer.count_z(inputs)
            inputs = layer.get_a(z)
        return inputs

    def teach_me(self, input_vector, output_vector):
        inputs = input_vector
        for layer in self.layers:
            z = layer.count_z(inputs)
            inputs = layer.get_a(z)

        cost = self.softmax_grad(inputs, output_vector) 
        for layer in reversed(self.layers):
            cost = layer.get_cost(cost)
