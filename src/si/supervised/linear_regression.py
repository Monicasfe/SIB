from .model import Model
import numpy as np
from src.si.util.util import mse

class LinearRegression(Model):

    def __init__(self, gd=False, epochs=1000, lr=0.001):
        super(LinearRegression, self).__init__()
        self.gd = gd
        self.theta = None
        self. epochs = epochs
        self.lr = lr

    def fit(self, dataset):
        X, Y = dataset.getXy()
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.X =X
        self.Y = Y
        #Close form or GD
        self.train_gd(X, Y) if self.gd else self.train_closed(X, Y)
        self.is_fitted = True

    def train_closed(self, X, Y):
        """
        theta = (X(T)X)**-1 * (X(T)Y)
        Uses closed form linear algebra top fit the model.
        """
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def train_gd(self, X, Y):
        m = X.shape[0]
        n = X.shape[1]
        self.history = {} #ignorar por enquanto
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            # (1/m) * (-Y(T) + theta(T)X(T)) * X ==> gradient
            grad = 1/m * (X.dot(self.theta)-Y).dot(X)
            self.theta -= self.lr * grad
            self.history[epoch] = [self.theta[:], self.cost()] #permite obter os graficos que estao no notebook para ver a veolucao do loss ao longo do tempo

    def predict(self, x):
        assert self.is_fitted, "Model must be fitted before predicting"
        x1 = np.hstack(([1], x))
        return np.dot(self.theta, x1)

    def cost(self):
        y_pred = np.dot(self.X, self.theta)
        return mse(self.Y, y_pred)/2


class LinearRegressionReg(LinearRegression): #Isto  com regularizacao

    def __init__(self, gd = False, epochs=1000, lr=0.001, lbd=1): #lbd = lambda
        """
        Linear Regression model with L2 regularization
        """
        super(LinearRegressionReg, self).__init__(gd=gd, epochs=epochs, lr=lr)
        self.lbd = lbd

    def train_closed(self, X, Y):
        """
        Uses closed form linear algebra to fit the model.
        theta = inv(XT*X+lbd*I)*XT*Y
        """
        n = X.shape[1]
        identity = np.eye(n) #matriz de identidade do tamanho do n de features
        identity[0, 0] = 0 #por cauda de bias tem que ter 0 na primeira posicao
        self.theta = np.linalg.inv(X.T.dot(X)+self.lbd*identity).dot(X.T).dot(Y)
        self.is_fitted = True

    def train_gd(self, X, Y):
        """
        Uses gradient to fit the model.
        """
        m = X.shape[0]
        n = X.shape[1]
        self.history = {}  # ignorar por enquanto
        self.theta = np.zeros(n)
        lbds  = np.full(m, self.lbd) #cria um vetor de tamanho m com os valores de lambda
        lbds[0] = 0 #adiciona um 0 na posica 1 do vetor de lambdas
        for epoch in range(self.epochs):
            #(-Y(T) + theta(T)X(T)) * X ==> gradient
            grad = 1/m * (X.dot(self.theta) - Y).dot(X)
            self.theta -= (self.lr/m) * (lbds+grad)
            self.history[epoch] = [self.theta[:], self.cost()]


