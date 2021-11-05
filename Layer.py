from helper import generate_random
import numpy


class Layer:
    def __init__(self, large, next_layer_n, activity_fun):
        self.weights = numpy.random.rand(next_layer_n, large) - 0.5
        self.activity_fun = activity_fun
        self.bias = generate_random(-0.1, 0.1, next_layer_n)
        self.a = numpy.empty(next_layer_n)
        self.prev_a = numpy.empty(next_layer_n)
        self.z = numpy.empty(next_layer_n)
        self.cost = None

    def count_z(self, input_data):
        self.prev_a = input_data
        z = numpy.array(self.weights).dot(input_data)
        self.z = numpy.add(z, self.bias)
        return z

    def get_a(self, z):
        self.a = self.activity_fun(z)
        return self.a

    def get_cost(self, upper_cost):
        self.cost = upper_cost * self.a.T * self.activity_fun(self.prev_a, derivative=True)
        return self.cost

    def upadte_weights(self , alfa, m):
        self.weights = self.weights - alfa*numpy.sum(numpy.dot(self.cost, self.prev_a.T))/m
        return self.weights

