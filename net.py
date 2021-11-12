import copy
import pickle
import numpy as np

from Layer import Layer

from helper import modify_train_y

class Net:
    def __init__(self, alpha, epok=10, batch=30):
        self.layers = []
        self.alpha = alpha
        self.epok = epok
        self.batch = batch


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
        return np.mean(teaching_data.T * -1*np.log(y + 1e-8))

    def softmax_grad(self, y, teaching_data):
        data_as_vec = np.array([teaching_data])
        diff = data_as_vec.T-y
        return -1*diff

    def forward(self, input_vector):
        inputs = np.array([input_vector]).T
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
        self.layers[-1].errors.append(copy.deepcopy(error))
        #print("error")
        #print(error)
        #print("aktualizacja wag (prev_a)")
        #print(self.layers[-1].prev_a)
        id=0
        for layer in reversed(self.layers):
            if id > 0:
                #print(self.layers[-id].weights.T)
                actvity = layer.activity_fun(layer.z, derivative=True)
                error = np.dot(self.layers[-id].weights.T, error)*actvity
                self.layers[-id-1].errors.append(copy.deepcopy(error))
            id = id + 1
        if np.argmax(inputs) == np.argmax(output_vector.T):
            return 1
        return 0

    def update_wieghts(self):
        id=1
        for layer in reversed(self.layers):
            weights_sum = np.zeros(shape=layer.weights.shape)
            for x in range(self.batch):
                weights_sum += np.dot(layer.errors[x], layer.outputs[x].T)
            layer.weights -= self.alpha/self.batch * weights_sum

            bias_sum = np.zeros(shape=layer.bias.shape)
            for x in range(self.batch):
                bias_sum += layer.errors[x]
            layer.bias -= self.alpha/self.batch * bias_sum #TOCHECK
            layer.errors=[]
            layer.outputs=[]
            id +=1

    def teach_me(self, learning_data, test_data):
        max_accuracy = 0
        for i in range(self.epok):
            sum_loss = 0
            id_of_x_in_pack = 1
            for k in range(len(learning_data[0])): #dla wszystkich wzorców
                x = np.ndarray.flatten(learning_data[0][k])
                sum_loss += self.sample(x, learning_data[1][k], len(learning_data))
                if id_of_x_in_pack == self.batch:
                    self.update_wieghts()
                    id_of_x_in_pack = 0
                id_of_x_in_pack +=1
            if sum_loss > max_accuracy:
                max_accuracy = sum_loss
                self.save_layer()

            print(i, sum_loss/len(learning_data[0]))
            self.test_mlp(test_data[0], test_data[1])

    def test_mlp(self, test_data_X, test_data_Y):
        correct=0
        for i in range(len(test_data_X)):
            x = np.ndarray.flatten(test_data_X[i])
            predict = self.forward(x)
            #print(self.forward(x), test_data_Y[i])
            if np.argmax(predict) == np.argmax(test_data_Y[i]):
                correct = correct + 1
        print(correct/len(test_data_Y))

    def save_layer(self):
        with open('layers.pkl', 'wb') as outp:
            pickle.dump(self.layers, outp, pickle.HIGHEST_PROTOCOL)

    def layers_from_file(self):
        with open('company_data.pkl', 'rb') as inp:
            self.layers = pickle.load(inp)


def mse(res, y):
    return np.mean(np.power(res-y, 2))


def mse_prime(res, y):
    return 2*(y-res)/res.size


