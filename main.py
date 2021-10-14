from perceptron import PerceptronAND

learnig_data = [
    ([1, 1], 1),
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
]


def run():
    perceptron = PerceptronAND([-0.2, 0.9], 0.1, 0.5, False)
    perceptron.teach_me(learnig_data)
    print(perceptron.classify([1, 1]))
    print(perceptron.classify([0, 0]))
    print(perceptron.classify([1, 0]))
    print(perceptron.classify([0, 1]))

    perceptron = PerceptronAND([0.3, -0.2, 0.9], 0.1, 0, True)
    perceptron.teach_me(learnig_data)
    print(perceptron.classify([1, 1]))
    print(perceptron.classify([0, 0]))
    print(perceptron.classify([1, 0]))
    print(perceptron.classify([0, 1]))


if __name__ == '__main__':
    run()
