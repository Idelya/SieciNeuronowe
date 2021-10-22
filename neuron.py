from helpers import generate_random


class Neuron:
    def __init__(self, activationFun, y=0, isBias=False):#activationFun - callback
        self.activationFun = activationFun
        self.weight = []
        self.y = y

    def generateWeight(self, l):
        self.weight = generate_random(-0.1, 0.1)

    def get_z(self, data):
        return sum(data)

    def get_a(self, z):
        return self.activationFun(z)

    def get_res(self, data):
        self.y = self.get_a(self.get_z(data))
        return self.y