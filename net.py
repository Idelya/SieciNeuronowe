import copy

import numpy as np

from Layer import Layer


class Net:
    def __init__(self):
        self.layers = []

    def configLayers(self, vector_length, config_layer, start_fun, exit_fun):
        self.layers.append(Layer(vector_length, config_layer[0]["l"], start_fun))
        for i in range(len(config_layer)):
            outputs_length = vector_length
            activation_fun = exit_fun
            if i + 1 < len(config_layer):
                outputs_length = config_layer[i + 1]["l"]
                activation_fun = config_layer[i]["activationFun"]
            self.layers.append(Layer(config_layer[i]["l"], outputs_length, activation_fun))

    def forward(self, input_vector):
        inputs = input_vector
        for layer in self.layers:
            z = layer.count_z(inputs)
            inputs = layer.get_a(z)
        return inputs
