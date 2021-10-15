from perceptron import Adaline
import numpy
import random

down_input = -1  # bipolar
learnig_data = [
    (numpy.array([1, 1]), 1),
    (numpy.array([1.1, 0.99]), 1),
    (numpy.array([0.89, 0.99]), 1),
    (numpy.array([1.123, 0.97]), 1),
    (numpy.array([0, 0]), down_input),
    (numpy.array([0.14, 0.16]), down_input),
    (numpy.array([0, 1]), down_input),
    (numpy.array([0.13, 0.97]), down_input),
    (numpy.array([1, 0]), down_input),
    (numpy.array([1.1, 0]), down_input),
    (numpy.array([1, 0.13]), down_input),
    (numpy.array([0.1, -0.1]), down_input),
]

random_range = -1


def generate_random(a, b, n):
    res = numpy.empty(n)
    for i in range(n):
        res[i] = random.uniform(a, b)
    return res


def run():
    weights = generate_random(-1 * random_range, random_range, 3)
    perceptron = Adaline(weights, 0.03, 0.28, 0,
                         down_input)  # weight, learning_rate, permissible_error, threshold, down_input):
    perceptron.teach_me(learnig_data)
    print(perceptron.classify(numpy.array([1, 1])))
    print(perceptron.classify(numpy.array([0, 0])))
    print(perceptron.classify(numpy.array([1, 0])))
    print(perceptron.classify(numpy.array([0, 1])))
    print(perceptron.classify(numpy.array([0.9, 0.9])))
    print(perceptron.classify(numpy.array([0.1, 0.9])))
    print(perceptron.classify(numpy.array([0.9, 0.1])))


if __name__ == '__main__':
    run()
