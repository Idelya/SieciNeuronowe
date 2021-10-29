from helper import generate_random
import numpy


class Layer:
    def __init__(self, large, next_layer_n, activity_fun):
        self.weights = numpy.random.rand(next_layer_n, large) - 0.5
        self.activity_fun = activity_fun
        self.bias = generate_random(-0.1, 0.1, next_layer_n)
        self.a = numpy.empty()
        self.z = numpy.empty()

    def count_z(self, input_data):
        z = numpy.array(self.weights).dot(input_data)
        self.z = numpy.add(z, self.bias)
        return z

    def get_a(self, z):
        self.a = self.activity_fun(z)
        return self.a

    def get_cost_of_previous(self, upper_cost):
        cost = numpy.dot(self.weights.T, upper_cost)
        return cost
