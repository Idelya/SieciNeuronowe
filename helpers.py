import random
import numpy


def generate_random(a, b, n):
    res = numpy.empty(n)
    for i in range(n):
        res[i] = random.uniform(a, b)
    return res
