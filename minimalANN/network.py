import numpy as np
from . import layer

class NeuralNetwork:
    def __init__(self):
        self.layers:list[layer.Layer] = []
        self.learning_rate = 0.1
        self.sensitivities = {}
        self.n = {} # n values of each layer
        self.a = {} # a values of each layer
        self.weights = {}
        self.bias = {}
        self.error = 0

    def train(self, X, y, epochs):
        from . import backpropagation as bp
        for _ in range(epochs):
            print(f"EPOCH : {_+1}\n\n\n")
            bp.forward(self, X, y)
            bp.calculate_sensitivities(self, X, y)
            bp.backprop(self, X, y)
        return self