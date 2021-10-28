from helper import generate_random
import numpy


class Layer:
    def __init__(self, large, next_layer_n, activity_fun):
        self.weights = numpy.random.rand(next_layer_n, large) - 0.5
        self.activity_fun = activity_fun
        self.bias = generate_random(-0.1, 0.1, next_layer_n)

    def count_z(self, input_data):
        z = numpy.array(self.weights).dot(input_data)
        return numpy.add(z, self.bias)

    def get_a(self, z):
        return self.activity_fun(z)
