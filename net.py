import copy

import numpy as np

from neuron import Neuron


class Net:
    def __init__(self, learn_data, layers_config):
        self.learn_data = learn_data
        self.layers = []
        self.layers_config = layers_config

    def configLayers(self, exitFun):
            entry_layer = []
            for x in self.learn_data:
                entry_layer.append(Neuron(lambda value: value, x))#warstwa wejścia nie ma funkcji aktywacji, zwraca value
            self.layers.append(entry_layer)

            layers_id = 1
            for conf in self.layers_config:
                neuronsOfLayer = []
                for n in self.layers[layers_id-1]:
                    n.generateWeight(conf["l"])

                for id in range(conf["l"]):
                    neuronsOfLayer.append(Neuron(conf["activationFun"], 0))

                self.layers.append(copy.deepcopy(neuronsOfLayer))
                layers_id = layers_id + 1

            for n in self.layers[layers_id-1]:
                n.generateWeight(len(self.learn_data))

            #ending layer
            exit_layer = []
            for _ in self.learn_data:
                exit_layer.append(Neuron(exitFun))
            self.layers.append(exit_layer)

    def getMatrix(self, layer):
        matrix = []
        for i in range(len(layer[0].weight)):
            vector_of_entries = np.empty(len(layer))
            id = 0
            for neuron in layer:
                vector_of_entries[id] = neuron.weight[i] * neuron.y
                id = id + 1
            matrix.append(vector_of_entries)
        return matrix

    def forward(self):
        matrix = 0
        result = []
        layer_id = 0
        for layer in self.layers:
            #dla każdej warstwy
            if layer_id != 0: #wejście
                i = 0
                for neuron in layer:
                    neuron.get_res(matrix[i])
                    i = i + 1
            matrix = self.getMatrix(layer)

            if layer_id == len(self.layers) - 1: #warstwa wyjścia
                for neuron in layer:
                    result.append(neuron.y)
            layer_id = layer_id + 1
        return result









