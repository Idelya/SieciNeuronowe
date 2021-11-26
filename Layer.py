import copy

from helper import generate_random
import numpy


class Layer:
    def __init__(self, large, next_layer_n, activity_fun):
        self.weights = []
        self.errors = []
        self.outputs = []
        for i in range(next_layer_n):
            self.weights.append(numpy.random.normal(0, 0.1, size=large))
        self.weights = numpy.array(self.weights)
        self.sum_of_grad = numpy.zeros(shape=numpy.shape(self.weights)) #adagrad
        self.E_grad = numpy.zeros(shape=numpy.shape(self.weights)) #adadelta
        self.E_theta = numpy.zeros(shape=numpy.shape(self.weights)) #adadelta
        self.m = numpy.zeros(shape=numpy.shape(self.weights)) #adam
        self.v = numpy.zeros(shape=numpy.shape(self.weights)) #adam
        self.activity_fun = activity_fun
        self.bias = numpy.array([numpy.random.normal(0, 0.1, size=next_layer_n)]).T
        self.sum_of_grad_bias = numpy.zeros(shape=numpy.shape(self.bias))#adagrad
        self.E_grad_b = numpy.zeros(shape=numpy.shape(self.bias)) #adadelta
        self.E_theta_b = numpy.zeros(shape=numpy.shape(self.bias)) #adadelta
        self.m_b = numpy.zeros(shape=numpy.shape(self.bias)) #adam
        self.v_b = numpy.zeros(shape=numpy.shape(self.bias)) #adam
        self.a = numpy.random.normal(scale=1.0, size=next_layer_n)
        self.prev_a = numpy.empty(large)
        self.z = numpy.empty(large)
        self.cost = None

    def count_z(self, input_data):
        self.outputs.append(copy.deepcopy(input_data))
        #print(self.weights)
        #print("dot")
        #print(input_data)
        self.prev_a = input_data
        z = self.weights.dot(input_data)
        #print(z)
        #print("add bias")
        #print(self.bias.T)
        self.z = numpy.add(z, self.bias)
        #print(self.z)
        return self.z

    def get_a(self, z):
        #print('count a:')
        self.a = self.activity_fun(z)
        #print(self.a)
        return self.a


