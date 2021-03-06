class PerceptronAND:
    def __init__(self, weight, alfa, threshold, with_bias, down_input):
        self.weight = weight
        self.threshold = threshold
        self.alfa = alfa
        self.with_bias = with_bias
        self.down_input = down_input

    def classify(self, input_data, with_bias = False):  # input_data = [x1, x2]
        if with_bias:
            input_vector = [1] + input_data
        else:
            input_vector = input_data

        z = sum([a * b for a, b in zip(self.weight, input_vector)])
        if z > self.threshold:
            return 1
        return self.down_input

    def teach_me(self, learning_data):
        if self.with_bias:
            test_data = list(([1] + data[0], data[1]) for data in learning_data)
        else:
            test_data = learning_data
        check = False
        epoka = 0
        while not check: #epoka
            print(epoka)
            check = True
            test_data_id = 0
            for data in test_data:
                classified = (self.classify(data[0]), data[1])
                if not classified[1] == classified[0]:
                    check = False
                    mistake = classified[1] - classified[0]
                    weight_as_list = list(self.weight)
                    new_weights = list()
                    for index in range(len(weight_as_list)):
                            current_weight = weight_as_list[index]
                            input_value = test_data[test_data_id][0][index]
                            new_weight = (current_weight + self.alfa * mistake * input_value)
                            new_weights.append(new_weight)
                    self.weight = new_weights
                test_data_id = test_data_id + 1
            epoka = 1 + epoka
        print(self.weight)
        return epoka
