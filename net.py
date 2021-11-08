import copy

import numpy as np

from Layer import Layer

from helper import modify_train_y

class Net:
    def __init__(self, alpha, epok=10):
        self.layers = []
        self.alpha = alpha
        self.epok = epok

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
        return -np.mean(teaching_data * np.log(y.T + 1e-8))

    def softmax_grad(self, y, teaching_data):
        data_as_vec = np.array([teaching_data])
        diff = y.T-data_as_vec
        return -1*diff

    def forward(self, input_vector):
        inputs = input_vector
        for layer in self.layers:
            z = layer.count_z(inputs)
            inputs = layer.get_a(z)
        return inputs

    def sample(self, input_vector, output_vector, m):
        inputs = np.array([input_vector]).T
        for layer in self.layers:
            z = layer.count_z(inputs)
            inputs = layer.get_a(z)
        #print('loss')
        sum_loss = self.loss(inputs, output_vector)
        #print(sum_loss)

        #cost = np.dot(self.layers[len(self.layers)-1].a.T, self.softmax_grad(inputs, output_vector))
        #cost = mse_prime(inputs, output_vector)
        #print("grad softmax ")
        #print(self.softmax_grad(inputs, output_vector))
        error = self.softmax_grad(inputs, output_vector)
        #print("error")
        #print(error)
        #print("aktualizacja wag (prev_a)")
        #print(self.layers[-1].prev_a)
        arr_errors = []
        arr_errors.append(copy.deepcopy(error))
        id=0
        for layer in reversed(self.layers):
            if id > 0:
                #print(self.layers[-id].weights.T)
                error = np.dot(self.layers[-id].weights.T, error.T).T
                arr_errors.append(copy.deepcopy(error))
            id = id + 1
        id=0
        self.layers[-1].weights -= self.alpha * arr_errors[0].T.dot(self.layers[-1].prev_a.T)#aktualizacja dla W3 - do sparwdzenia kolejnośc mnożenia
        self.layers[-1].bias -= self.alpha * arr_errors[0]
        print(self.layers[-3].weights)
        for layer in reversed(self.layers):
            if id > 0:
                error = arr_errors[id]
                actvity = layer.activity_fun(layer.prev_a, derivative=True)
                self.layers[-id-1].weights -= self.alpha * error.T.dot(actvity.T)
                self.layers[-id-1].bias -= self.alpha * error
            id = id + 1
        return sum_loss

    def teach_me(self, learning_data):
        for i in range(self.epok):
            sum_loss = 0
            for k in range(len(learning_data[0])):
                x = np.ndarray.flatten(learning_data[0][k])
                sum_loss += self.sample(x, learning_data[1][k], len(learning_data))

            #print(i, sum_loss/len(learning_data[0]))

    def test_mlp(self, test_data_X, test_data_Y):
        correct=0
        for i in range(len(test_data_X)):
            x = np.ndarray.flatten(test_data_X[i])
            predict = self.forward(x)
            #print(self.forward(x), test_data_Y[i])
            if np.argmax(predict) == np.argmax(test_data_Y[i]):
                correct = correct + 1
        #print(correct/len(test_data_Y))


def mse(res, y):
    return np.mean(np.power(res-y, 2))


def mse_prime(res, y):
    return 2*(y-res)/res.size


