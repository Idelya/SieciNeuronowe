import random
import numpy


def generate_random(a, b, n):
    res = numpy.empty(n)
    for i in range(n):
        res[i] = random.uniform(a, b)
    return res


def sig(z):
    return 1 / (1 + numpy.power(numpy.e, -z))


def sig_vec(z):
    for i in range(len(z)):
        z[i] = sig(z[i])
    return z


def tanh(z):
    return numpy.tanh(z)


def tanh_vec(z):
    for i in range(len(z)):
        z[i] = tanh(z[i])
    return z


def relu(z):
    if z < 0:
        return 0
    else:
        return z


def relu_vec(z):
    for i in range(len(z)):
        z[i] = relu(z[i])
    return z


def softmax(x):
    return numpy.exp(x) / numpy.sum(numpy.exp(x))
