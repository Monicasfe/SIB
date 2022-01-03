import numpy as np
from src.si.supervised.model import Model
from scipy import signal
from abc import ABC, abstractmethod
from src.si.util.util import mse, mse_prime



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
        #compute rhe weights error dE/dW = X.T * cE/dY
        weights_error = np.dot(self.input.T, output_error)
        #compute the bias error dE/dB = dE/dY
        bias_error = np.sum(output_error, axis=0)
        #error dE/dX to pass on to the previous layer
        input_error = np.dot(output_error, self.weights.T)
        #update parameters
        self.weights -= lr * weights_error
        self.bias -= lr * bias_error
        return input_error


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
        #lr is not used because there is no "learnable" parameters
        #only passed the error to the previous error
        return np.multiply(self.func.prime(self.input), output_error)


class NN(Model):

    def __init__(self, lr=0.1, epochs=100, verbose=True):
        """
        Neural Network model. the default function is MSE.
        :param epochs: number of epochs
        :param lr: learning rate
        """
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.layers = []
        self.loss = mse
        self.loss_prime = mse_prime

    def add(self, layer):
        """Add a layer to the NN"""
        self.layers.append(layer)

    def fit(self, dataset):
        X, Y = dataset.getXy()
        self.dataset = dataset
        self.history = dict()
        for epoch in range(self.epochs):
            output = X
            #forward propagation
            for layer in self.layers:
                output = layer.forward(output)

            #back propagation
            error = self.loss_prime(Y, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, self.lr)

            #calculate average error on all samples
            err = self.loss(Y, output)
            self.history[epoch] = err
            if self.verbose:
                print(f"epoch {epoch+1}/{self.epochs} error = {err}")
        if not self.verbose:
            print(f"error = {err}")

        self.is_fitted = True


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