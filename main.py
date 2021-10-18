from perceptron import PerceptronAND
import random

down_input = 0 #polar | bipolar
learnig_data = [
    ([1, 1], 1),
    ([1.1, 0.99], 1),
    ([0.89, 0.99], 1),
    ([1.123, 0.97], 1),
    ([0, 0], down_input),
    ([0.14, 0.16], down_input),
    ([0, 1], down_input),
    ([0.23, 0.97], down_input),
    ([1, 0.23], down_input),
    ([1.1, 0], down_input),
    ([0.1, -0.1], down_input),
]
random_range = 0.5

def generate_random(a, b, n):
    res = list()
    for i in range(n):
        res.append(random.uniform(a, b))
    return res

def run():
    n = 100
    epok = 0
    correctness = 0
    for i in range(n):
        print(i)

        weights = generate_random(-1 * random_range, random_range, 3)
        perceptron = PerceptronAND(weights, 4, 0, True, down_input)
        epok = perceptron.teach_me(learnig_data) + epok
        if(perceptron.classify([1, 1], True) == 1):
            correctness = correctness + 1

        if(perceptron.classify([0, 0], True) == 0):
            correctness = correctness + 1

        if (perceptron.classify([1, 0], True) == 0):
            correctness = correctness + 1

        if (perceptron.classify([0, 1], True) == 0):
            correctness = correctness + 1

        if (perceptron.classify([1.1, 0.98], True) == 1):
            correctness = correctness + 1


    print("result\n")
    print(epok/n)
    print(correctness/(n*5))
    #weights = generate_random(-1*random_range, random_range, 3)
    #perceptron_b = PerceptronAND(weights, 0.1, 0, True, down_input)
    #perceptron_b.teach_me(learnig_data)
    #print(perceptron_b.classify([1, 1], True))
    #print(perceptron_b.classify([0, 0], True))
    #print(perceptron_b.classify([1, 0], True))
    #print(perceptron_b.classify([0, 1], True))
    #print(perceptron_b.classify([0.97, 1.1], True))


if __name__ == '__main__':
    run()
