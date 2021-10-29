import random
import numpy


def generate_random(a, b, n):
    res = numpy.empty(n)
    for i in range(n):
        res[i] = random.uniform(a, b)
    return res


def sig(z):
    return 1 / (1 + numpy.power(numpy.e, -z))


def sig_derivative(z):
    return (numpy.power(numpy.e, z))/(numpy.power((numpy.power(numpy.e, z) + 1), 2))


def sig_vec(z, derivative=False):
    fun = sig
    if derivative:
        fun = sig_derivative
    for i in range(len(z)):
        z[i] = fun(z[i])
    return z


def tanh(z):
    return numpy.tanh(z)

def tanh_derivative(z):
    return 1 - numpy.tanh(z) ** 2

def tanh_vec(z, derivative=False):
    fun = tanh
    if derivative:
        fun = tanh_derivative
    for i in range(len(z)):
        z[i] = fun(z[i])
    return z


def relu(z):
    if z < 0:
        return 0
    else:
        return z


def relu_derivative(z):
    if z < 0:
        return 0
    else:
        return 1


def relu_vec(z, derivative=False):
    fun = relu
    if derivative:
        fun = relu_derivative
    for i in range(len(z)):
        z[i] = fun(z[i])
    return z


def softmax(x):
    return numpy.exp(x) / numpy.sum(numpy.exp(x))