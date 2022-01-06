import numpy as np
from src.si.supervised.model import Model
from abc import ABC, abstractmethod
from src.si.util.util import mse, mse_prime
from src.si.util.im2col import pad2D, im2col, col2im



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
        self.weights = np.random.rand(input_size, output_size) - 0.5
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

    def __init__(self, lr=0.01, epochs=1000, verbose=True):
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
            else:
                print(f"epoch {epoch+1}/{self.epochs} error = {err}", end="\r")
        # if not self.verbose:
        #     print(f"error = {err}")

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
        Y = Y if Y is not None else self.dataset.Y
        output = self.predict(X)
        return self.loss(Y, output)


class Flatten(Layer):

    def forward(self, input_v):
        self.input_shape = input_v.shape
        output = input_v.reshape(input_v.shape[0], -1)
        return output

    def backward(self, output_error, lr):
        return output_error.reshape(self.input_shape)


class Conv2D(Layer):

    def __init__(self, input_shape, kernel_shape, layer_depth, stride=1, padding=0):
        self.input_shape = input_shape
        self.in_ch = input_shape[2]
        self.out_ch = layer_depth
        self.stride = stride
        self.padding = padding
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], self.in_ch, self.out_ch) - 0.5
        self.bias = np.zeros((self.out_ch, 1))


    def forward(self, input_v):
        s = self.stride
        self.X_shape = input_v.shape
        _, p = pad2D(input_v, self.padding, self.weights.shape[:2], s)
        pr1, pr2, pc1, pc2 = p
        fr, fc, in_ch, out_ch = self.weights.shape
        n_ex, in_rows, in_cols, in_ch = input_v.shape

        #compute the dimensions of the convolution output
        out_rows = int((in_rows + pr1 + pr2 - fr) / s + 1)
        out_cols = int((in_cols + pc1 + pc2 - fc) / s + 1)

        #convert X and W into the appropriate 2D matrices and take their product
        self.X_col, _ = im2col(input_v, self.weights.shape, p, s)
        W_col = self.weights.transpose(3, 2, 0, 1).reshape(out_ch, -1)

        output_data = (W_col @ self.X_col + self.bias).reshape(out_ch, out_rows, out_cols, n_ex).transpose(3, 1, 2, 0)

        return output_data

    def backward(self, output_error, lr):
        fr, fc, in_ch, out_ch = self.weights.shape
        p = self.padding
        db = np.sum(output_error, axis=(0, 1, 2))
        db = db.reshape(out_ch,)
        dout_reshape = output_error.transpose(1, 2, 3, 0).reshape(out_ch, -1)
        dW = dout_reshape @ self.X_col.T
        dW = dW.reshape(self.weights.shape)

        W_reshape = self.weights.reshape(out_ch, -1)
        dX_dol = W_reshape.T @ dout_reshape
        input_error = col2im(dX_dol, self.X_shape, self.weights.shape, (p, p, p, p), self.stride)

        self.weights -= lr * dW
        self.bias -= lr * db

        return input_error