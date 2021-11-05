from helper import generate_random
import numpy


class Layer:
    def __init__(self, large, next_layer_n, activity_fun):
        self.weights = []
        for i in range(large):
            self.weights.append(numpy.random.normal(scale=1.0, size=next_layer_n))
        self.activity_fun = activity_fun
        self.bias = numpy.random.normal(scale=1.0, size=next_layer_n)
        self.a = numpy.random.normal(scale=1.0, size=next_layer_n)
        self.prev_a = numpy.empty(large)
        self.z = numpy.empty(large)
        self.cost = None

    def count_z(self, input_data):
        self.prev_a = input_data
        z = numpy.array(input_data).dot(self.weights)
        self.z = numpy.add(z, self.bias)
        return z

    def get_a(self, z):
        self.a = self.activity_fun(z)
        return self.a

    def get_cost(self, upper_cost):
        #self.cost = upper_cost * self.a.T * self.activity_fun(self.prev_a, derivative=True)
        self.cost = upper_cost*self.activity_fun(self.prev_a, derivative=True)
        return self.cost

    def get_cost_with_weight(self, upper_cost):
        self.cost = self.get_cost(upper_cost.dot(numpy.array(self.weights).T))
        return self.cost

    def upadte_weights(self, alfa):
        self.weights -= alfa * numpy.dot(self.prev_a.T, self.cost)
        self.bias -= alfa * self.cost

