import numpy

class Adaline:
    def __init__(self, weight, learning_rate, permissible_error, threshold, down_input):
        self.weight = weight
        self.learning_rate = learning_rate
        self.permissible_error = permissible_error
        self.threshold = threshold
        self.down_input = down_input

    def classify(self, input_data):
        input_vector = numpy.insert(numpy.array(input_data), 0, 1, axis=0)
        z = self.get_z(input_vector)
        return self.get_threshold(z)

    def get_z(self, input_data):  # input_data = [x1, x2]
        z = numpy.array(input_data).dot(self.weight)
        return z

    def get_threshold(self, z):
        if z > self.threshold:
            return 1
        return self.down_input

    def teach_me(self, learning_data):
        test_data = list((numpy.insert(data[0], 0, 1, axis=0), data[1]) for data in learning_data)
        check = False
        epoka = 0
        while not check: #epoka
            print(epoka)
            mistake_sum = 0
            for data in test_data:
                z = self.get_z(data[0])
                mistake = pow(data[1] - z, 2)
                mistake_sum = mistake_sum + mistake
                for index in range(len(self.weight)):
                    self.weight[index] = self.weight[index] + 2 * self.learning_rate * (data[1] - z) * data[0][index]
            if mistake_sum/len(learning_data) < self.permissible_error:
                check = True
            epoka = 1 + epoka
        print(self.weight)
        print(mistake_sum/len(learning_data))
        return epoka