import copy
import math

from helper import generate_random
import numpy


class Layer:
    def __init__(self, large, next_layer_n, activity_fun, init_weight_method):
        self.weights = []
        self.errors = []
        self.outputs = []
        if init_weight_method=="normal":
            self.init_w_normal(next_layer_n, large)
        elif init_weight_method=="Xaviera":
            self.init_Xaviera(next_layer_n, large)
        elif init_weight_method=="He":
            self.init_He(next_layer_n, large)
        self.weights = numpy.array(self.weights)
        self.sum_of_grad = numpy.zeros(shape=numpy.shape(self.weights)) #adagrad
        self.E_grad = numpy.zeros(shape=numpy.shape(self.weights)) #adadelta
        self.E_theta = numpy.zeros(shape=numpy.shape(self.weights)) #adadelta
        self.m = numpy.zeros(shape=numpy.shape(self.weights)) #adam
        self.v = numpy.zeros(shape=numpy.shape(self.weights)) #adam
        self.activity_fun = activity_fun
        self.sum_of_grad_bias = numpy.zeros(shape=numpy.shape(self.bias))#adagrad
        self.E_grad_b = numpy.zeros(shape=numpy.shape(self.bias)) #adadelta
        self.E_theta_b = numpy.zeros(shape=numpy.shape(self.bias)) #adadelta
        self.m_b = numpy.zeros(shape=numpy.shape(self.bias)) #adam
        self.v_b = numpy.zeros(shape=numpy.shape(self.bias)) #adam
        self.a = numpy.random.normal(scale=1.0, size=next_layer_n)
        self.prev_a = numpy.empty(large)
        self.z = numpy.empty(large)
        self.cost = None

    def init_w_normal(self, next_layer_n, large):
        standard_dev=0.3
        #init weights
        for i in range(next_layer_n):
            self.weights.append(numpy.random.normal(0, standard_dev, size=large))

        #init bias
        self.bias = numpy.array([numpy.random.normal(0, standard_dev, size=next_layer_n)]).T


    def init_Xaviera(self, next_layer_n, large):
        standard_dev = math.sqrt(2/(next_layer_n + large))
        #init weights
        for i in range(next_layer_n):
            self.weights.append(numpy.random.normal(0, standard_dev, size=large))

        #init bias
        self.bias = numpy.array([numpy.random.normal(0, standard_dev, size=next_layer_n)]).T


    def init_He(self, next_layer_n, large):
        standard_dev = math.sqrt(2 / large)
        #init weights
        for i in range(next_layer_n):
            self.weights.append(numpy.random.normal(0, standard_dev, size=large))

        #init bias
        self.bias = numpy.array([numpy.random.normal(0, standard_dev, size=next_layer_n)]).T

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


