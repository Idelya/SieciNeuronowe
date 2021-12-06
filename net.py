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
        self.with_optimalization = False
        self.method_config = {}
        self.t = 0

        self.x_axis = []
        self.y_axis = []
        self.x_acc_t_axis = []
        self.y_acc_t_axis = []
        self.x_acc_v_axis = []
        self.y_acc_v_axis = []

    def optimalizationOfAlpha(self, config):
        self.with_optimalization = True
        self.method_config = config
        if self.method_config["name"] == "momentum" or self.method_config["name"] == "momentum_nesterova":
            for layer in self.layers:
                self.method_config["last_update"].append(np.zeros(shape=layer.weights.shape))
                self.method_config["last_update_bias"].append(np.zeros(shape=layer.bias.shape))

    def configLayers(self, vector_length, config_layer, start_fun, exit_fun, class_amount, init_weight_method):
        self.layers.append(Layer(vector_length, config_layer[0]["l"], start_fun, init_weight_method))
        for i in range(len(config_layer)):
            outputs_length = class_amount
            activation_fun = exit_fun
            if i + 1 < len(config_layer):
                outputs_length = config_layer[i + 1]["l"]
                activation_fun = config_layer[i]["activationFun"]
            self.layers.append(Layer(config_layer[i]["l"], outputs_length, activation_fun, init_weight_method))

    def loss(self, y, teaching_data):
        return 1 - np.mean(-1 * np.log(y + 1e-8) * teaching_data.T)

    def softmax_grad(self, y, teaching_data):
        data_as_vec = np.array([teaching_data])
        diff = data_as_vec.T - y
        return -1 * diff

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
        # print('loss')
        sum_loss = self.loss(inputs, output_vector)
        # print(sum_loss)

        # cost = np.dot(self.layers[len(self.layers)-1].a.T, self.softmax_grad(inputs, output_vector))
        # cost = mse_prime(inputs, output_vector)
        # print("grad softmax ")
        # print(self.softmax_grad(inputs, output_vector))
        error = self.softmax_grad(inputs, output_vector)
        self.layers[-1].errors.append(copy.deepcopy(error))
        # print("error")
        # print(error)
        # print("aktualizacja wag (prev_a)")
        # print(self.layers[-1].prev_a)
        id = 0
        for layer in reversed(self.layers):
            if id > 0:
                # print(self.layers[-id].weights.T)
                actvity = layer.activity_fun(layer.z, derivative=True)

                # ================== momentum ===============
                if self.with_optimalization and self.method_config["name"] == "momentum_nesterova":
                    error = np.dot(
                        (self.layers[-id].weights - self.method_config["factor"] * self.method_config["last_update"][-id]).T, error) * actvity
                else:
                    error = np.dot(self.layers[-id].weights.T, error) * actvity
                self.layers[-id - 1].errors.append(copy.deepcopy(error))
            id = id + 1
        if np.argmax(inputs) == np.argmax(output_vector.T):
            return 1, sum_loss
        return 0, sum_loss

    def update_wieghts(self):
        id = 1
        for layer in reversed(self.layers):
            weights_sum = np.zeros(shape=layer.weights.shape)
            for x in range(self.batch):
                weights_sum += np.dot(layer.errors[x], layer.outputs[x].T)
            # ================== momentum ===============
            if self.with_optimalization and self.method_config["name"] == "momentum" or self.method_config[
                "name"] == "momentum_nesterova":
                diff = self.method_config["factor"] * self.method_config["last_update"][
                    -id] + self.alpha / self.batch * weights_sum
                layer.weights -= diff
                self.method_config["last_update"][-id] = diff
            # ================== adagrad   ===============
            elif self.with_optimalization and self.method_config["name"] == "adagrad":
                layer.sum_of_grad += weights_sum ** 2
                diff = self.alpha / (self.batch * np.sqrt(layer.sum_of_grad + 1e-8)) * weights_sum
                layer.weights -= diff
            # ================== adadelta   ===============
            elif self.with_optimalization and self.method_config["name"] == "adadelta":
                layer.E_grad = self.method_config["factor"]*layer.E_grad + (1 - self.method_config["factor"])*(weights_sum ** 2)
                delta = (-1*(np.sqrt(layer.E_theta + 1e-8)/np.sqrt(layer.E_grad + 1e-8))*weights_sum)
                layer.E_theta = self.method_config["factor"]*layer.E_theta + (1 - self.method_config["factor"])*(delta ** 2)
                layer.weights += delta
            # ================== adam       ===============
            elif self.with_optimalization and self.method_config["name"] == "adam":
                layer.m = self.method_config["beta1"] * layer.m + (1 - self.method_config["beta1"]) * weights_sum
                layer.v = self.method_config["beta2"] * layer.v + (1. - self.method_config["beta2"]) * weights_sum ** 2
                corrected_m = layer.m / (1. - self.method_config["beta1"] ** (self.t+1))
                corrected_v = layer.v / (1. - self.method_config["beta2"] ** (self.t+1))
                diff = (self.alpha * corrected_m) / (np.sqrt(corrected_v) + 1e-8)
                layer.weights -= diff/self.batch
            else:
                layer.weights -= self.alpha/self.batch * weights_sum

            bias_sum = np.zeros(shape=layer.bias.shape)
            for x in range(self.batch):
                bias_sum += layer.errors[x]

            # ================== momentum ===============
            if self.with_optimalization and self.method_config["name"] == "momentum" or self.method_config[
                "name"] == "momentum_nesterova":
                diff = self.method_config["factor"] * self.method_config["last_update_bias"][
                    -id] + self.alpha / self.batch * bias_sum
                layer.bias -= diff
                self.method_config["last_update_bias"][-id] = diff
            # ================== adagrad   ===============
            elif self.with_optimalization and self.method_config["name"] == "adagrad":
                layer.sum_of_grad_bias += bias_sum ** 2
                diff = self.alpha / (self.batch * np.sqrt(layer.sum_of_grad_bias + 1e-8)) * bias_sum
                layer.bias -= diff
            # ================== adadelta   ===============
            elif self.with_optimalization and self.method_config["name"] == "adadelta":
                layer.E_grad_b = self.method_config["factor"]*layer.E_grad_b + (1 - self.method_config["factor"])*(bias_sum ** 2)
                delta = (-1*(np.sqrt(layer.E_theta_b + 1e-8)/np.sqrt(layer.E_grad_b + 1e-8))*bias_sum)
                layer.E_theta_b = self.method_config["factor"]*layer.E_theta_b + (1 - self.method_config["factor"])*(delta ** 2)
                layer.bias += delta
            # ================== adam   ===============
            elif self.with_optimalization and self.method_config["name"] == "adam":
                layer.m_b = self.method_config["beta1"] * layer.m_b + (1 - self.method_config["beta1"]) * bias_sum
                layer.v_b = self.method_config["beta2"] * layer.v_b + (1. - self.method_config["beta2"]) * bias_sum ** 2
                corrected_m = layer.m_b / (1. - self.method_config["beta1"] ** (self.t+1))
                corrected_v = layer.v_b / (1. - self.method_config["beta2"] ** (self.t+1))
                diff = (self.alpha * corrected_m) / (np.sqrt(corrected_v) + 1e-8)
                layer.bias -= diff/ self.batch
            else:
                layer.bias -= self.alpha / self.batch * bias_sum  # TOCHECK
            layer.errors = []
            layer.outputs = []
            id += 1
        self.t += 1

    def teach_me(self, learning_data, test_data):
        max_accuracy = 0
        for i in range(self.epok):
            sum_loss = 0
            accuracy = 0
            id_of_x_in_pack = 1
            for k in range(len(learning_data[0])):  # dla wszystkich wzorców
                x = np.ndarray.flatten(learning_data[0][k])
                res = self.sample(x, learning_data[1][k], len(learning_data))
                accuracy = accuracy + res[0]
                sum_loss = sum_loss + max(0, res[1])
                if id_of_x_in_pack == self.batch:
                    self.update_wieghts()
                    id_of_x_in_pack = 0
                id_of_x_in_pack += 1
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                self.save_layer()

            self.x_axis.append(i)
            self.y_axis.append(sum_loss / len(learning_data[0]))
            print(i, sum_loss / len(learning_data[0]))
            print(i, accuracy / len(learning_data[0]))
            self.x_acc_v_axis.append(i)
            self.y_acc_v_axis.append(accuracy / len(learning_data[0]))

            self.x_acc_t_axis.append(i)
            self.test_mlp(test_data[0], test_data[1])

    def test_mlp(self, test_data_X, test_data_Y):
        correct = 0
        for i in range(len(test_data_X)):
            x = np.ndarray.flatten(test_data_X[i])
            predict = self.forward(x)
            # print(self.forward(x), test_data_Y[i])
            if np.argmax(predict) == np.argmax(test_data_Y[i]):
                correct = correct + 1
        self.y_acc_t_axis.append(correct / len(test_data_Y))
        print(correct / len(test_data_Y))

    def save_layer(self):
        with open('layers.pkl', 'wb') as outp:
            pickle.dump(self.layers, outp, pickle.HIGHEST_PROTOCOL)

    def layers_from_file(self):
        with open('company_data.pkl', 'rb') as inp:
            self.layers = pickle.load(inp)


def mse(res, y):
    return np.mean(np.power(res - y, 2))


def mse_prime(res, y):
    return 2 * (y - res) / res.size
