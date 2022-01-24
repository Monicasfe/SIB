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
        super(Dense).__init__()
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
        super(Activation).__init__()
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
        super(NN).__init__()
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
            # else:
                # print(f"epoch {epoch+1}/{self.epochs} error = {err}", end="\r")
        if not self.verbose:
            print(f" epoch {self.epochs}/{self.epochs} error = {err}")

        self.is_fitted = True

    def fit_batch(self, dataset, batchsize=256):
        X, y = dataset.getXy()
        if batchsize > X.shape[0]:
            raise Exception('Number of batchs superior to length of dataset')
        n_batches = int(np.ceil(X.shape[0] / batchsize))
        self.dataset = dataset
        self.history = dict()
        for epoch in range(self.epochs):
            self.history_batch = np.zeros((1, batchsize))
            for batch in range(n_batches):
                output = X[batch * batchsize:(batch + 1) * batchsize, ]
                for layer in self.layers:
                    output = layer.forward(output)
                error = self.loss_prime(y[batch * batchsize:(batch + 1) * batchsize, ], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, self.lr)
                # calcule average error
                err = self.loss(y[batch * batchsize:(batch + 1) * batchsize, ], output)
                self.history_batch[0, batch] = err
            self.history[epoch] = np.average(self.history_batch)
            if self.verbose:
                print(f'epoch {epoch + 1}/{self.epochs}, error = {self.history[epoch]}')
            # else:
            #     print(f"epoch {epoch + 1}/{self.epochs}, error = {self.history[epoch]}", end='\r')
        if not self.verbose:
            print(f" epoch {self.epochs}/{self.epochs} error = {err}")
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

    def forward(self, input_shape):
        self.input_shape = input_shape.shape
        output = input_shape.reshape(input_shape.shape[0], -1)
        return output

    def backward(self, output_error, lr):
        return output_error.reshape(self.input_shape)


class Conv2D(Layer):

    def __init__(self, input_shape, kernel_shape, layer_depth, stride=1, padding=0):
        super(Conv2D).__init__()
        self.input_shape = input_shape
        self.in_ch = input_shape[2]
        self.out_ch = layer_depth
        self.stride = stride #e quanto move a janela ao longo do array
        self.padding = padding
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], self.in_ch, self.out_ch) - 0.5
        self.bias = np.zeros((self.out_ch, 1))


    def forward(self, input_shape):
        s = self.stride
        self.X_shape = input_shape.shape
        _, p = pad2D(input_shape, self.padding, self.weights.shape[:2], s)
        pr1, pr2, pc1, pc2 = p
        fr, fc, in_ch, out_ch = self.weights.shape
        n_ex, in_rows, in_cols, in_ch = input_shape.shape

        #compute the dimensions of the convolution output
        out_rows = int((in_rows + pr1 + pr2 - fr) / s + 1)
        out_cols = int((in_cols + pc1 + pc2 - fc) / s + 1)

        #convert X and W into the appropriate 2D matrices and take their product
        self.X_col, _ = im2col(input_shape, self.weights.shape, p, s)
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

class Pooling2D(Layer):

    def __init__(self, size=2, stride=2):
        super(Pooling2D).__init__()
        self.size = size
        self.stride = stride

    def pool(self, X_col):
        raise NotImplementedError

    def dpool(self, dX_col, dout_col, pool_cache):
        raise NotImplementedError

    def forward(self, input):
        print(self.size)
        self.X_shape = input.shape
        n, h, w, d = input.shape #n de imagens, comprimento, largura, n de canais
        h_out = (h - self.size) / self.stride + 1
        w_out = (w - self.size) / self.stride + 1

        if not w_out.is_integer() or not h_out.is_integer():
            raise Exception("Invalid output dimenstion")

        h_out, w_out = int(h_out), int(w_out)
        X_reshaped = input.reshape(n * d, h, w, 1)
        self.X_col, _ = im2col(X_reshaped, (self.size, self.size, d, d), pad=0, stride=self.stride) #nao da porque tem mais argumentos que a func que prof forneceu

        out, self.max_idx = self.pool(self.X_col)
        out = out.reshape(h_out, w_out, n, d)
        # out = out.transpose(3, 2, 0, 1)
        out = out.transpose(0, 1, 2, 3)

        return out


    def backward(self, output_error, lr):
        n, w, h, d = self.X_shape
        dX_col = np.zeros_like(self.X_col)
        dout_col = output_error.transpose(1, 2, 3, 0).ravel()

        dX = col2im(dX_col, (n * d , h, w, 1), self.size, pad=0, stride=self.stride)
        dX = self.dpool(dX_col, dout_col, self.max_idx)
        dX = dX.reshape(self.X_shape)

        return dX

    ##OLD FORWARD MINE
    # pool = self.pool_size
    # self.input = input_shape
    # h, w = input_shape.shape
    # output = np.zeros((h // pool, w // pool))
    #
    # for img_reg, i, j in self.iterate_pool(input_shape):
    #     output[i, j] = np.amax(img_reg, axis=(0, 1))
    #
    # return output
    #
    # def iterate_pool(self, input_img):
    #     size = self.size
    #     new_h = self.h // size
    #     new_w = self.w // size
    #     for i in range(new_h):
    #         for j in range(new_w):
    #             img_region = input_img[(i * size):(i * size + size), (j * size):(j * size + size)]
    #             yield img_region, i, j
    #

class MaxPooling2D(Pooling2D):

    def pool(self, X_col):
        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.size)]
        return out, max_idx

    def dpool(self, dX_col, dout_col, pool_cache):
        dX_col[pool_cache, range(dout_col.size)] = dout_col
        return dX_col


class RNN(Layer):

    def forward(self, input, a_prev, parameters):
        raise NotImplementedError

    def backward(self, output_error, lr):
        raise NotImplementedError
##implementar uma RNN