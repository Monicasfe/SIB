import numpy as np
from src.si.supervised.model import Model
from scipy import signal
from abc import ABC, abstractmethod
from src.si.util.util import mse



class Layer(ABC):

    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input_v):
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_error, lr):
        raise NotImplementedError


class Dense(Layer):

    def __init__(self, input_size, output_size):
        """Fully connecetd layer"""
        self.weights = np.random.rand(input_size, output_size)
        #temos tantos bias como o tamnho do output
        self.bias = np.random.rand(1,output_size)
    ##criar uma matriz de pesos

    def setWeights(self, weights, bias): #refazer os pesos
        if(weights.shape != self.weights.shape):
            raise ValueError (f"Shapes mismatch {weights.shape} and {self.weights.shape}")
        if (bias.shape != self.bias.shape):
            raise ValueError (f"Shapes mismatch {bias.shape} and {self.bias.shape}")
        self.weights = weights
        self.bias = bias

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, lr):
        raise NotImplementedError

class Activation(Layer):

    def __init__(self, reg_func):
        """
        Activation layer
        :param reg_func: function to use for regularization
        """
        self.func = reg_func

    def forward(self, input_data):
        self.input = input_data
        self.output = self.func(self.input)
        return self.output

    def backward(self, output_error, lr):
        raise NotImplementedError

class NN(Model):

    def __init__(self, lr=0.1, epochs=100, verbose=True):
        """
        Neural Network model
        :param epochs: number of epochs
        :param lr: learning rate
        :param verbose:
        """
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.layers = []
        self.loss = mse
        # self.loss_prime = mse_prime

    def add(self, layer):
        """Add a layer to the NN"""
        self.layers.append(layer)

    def fit(self, dataset):
        raise NotImplementedError

    def predict(self, input_data):
        assert self.is_fitted, "Model must be fitted"
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def cost(self, X=None, Y=None):
        assert self.is_fitted, "Model must be fitted"
        X = X if X is not None else self.dataset.X
        Y= Y if Y is not None else self.dataset.Y
        output = self.predict(X)
        return self.loss(Y, output)