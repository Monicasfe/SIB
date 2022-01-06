from .model import Model
from src.si.util.util import sigmoid, add_intersect
import numpy as np

class LogisticRegression(Model):

    def __init__(self, gd=False, epochs=1000, lr=0.1):
        """
        Logistic regression model
        :param gd: If true uses gradient descent (GD) to train the model.
            If False uses the closed from linear algebra. Default is False.
        :param epochs: Number of epochs for gd
        :param lr: learning rate for gd
        """
        super(LogisticRegression, self).__init__()
        self.gd = gd
        self.theta = None
        self.epochs = epochs
        self.lr = lr


    def fit(self, dataset):
        X , Y = dataset.getXy()
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.X = X
        self.Y = Y
        self.train(X, Y)
        self.is_fitted = True

    def train(self, X, Y):
        n = X.shape[1]
        self.history = {}
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            z = np.dot(X, self.theta)
            h = sigmoid(z)
            gradient = np.dot(X.T, (h - Y)) / Y.size
            self.theta -= self.lr * gradient
            self.history[epoch] = [self.theta[:], self.cost()]

    def probability(self, X):
        assert self.is_fitted, "Model must be fitted before predicting"
        x = np.hstack(([1], X))
        return sigmoid(np.dot(self.theta, x))


    def predict(self, x):
        x = np.array(x)
        if x.ndim > 1:
            res = []
            for i in x:
                p = self.probability(i)
                pred = 1 if p >= 0.5 else 0
                res.append(pred)
        else:
            p = self.probability(x)
            res = 1 if p >= 0.5 else 0
        return res

    def cost(self, X=None, Y=None, theta=None ):
        # x = X if X is not None else self.X
        # y = Y if Y is not None else self.Y
        # theta1 = theta if theta is not None else self.theta
        X = add_intersect(X) if X is not None else self.X
        Y = Y if Y is not None else self.Y
        theta = theta if theta is not None else self.theta

        h = sigmoid(np.dot(X, theta))
        cost = (-Y * np.log(h) - (1-Y) * np.log(1-h))
        res = np.sum(cost) / X.shape[0]
        return res

class LogisticRegressionReg(LogisticRegression):
    """
    Linear regression model with L2 regularization
    """
    def __init__(self, gd=False, epochs=1000, lr=0.1, lbd=1):

        super(LogisticRegressionReg, self).__init__(gd=gd, epochs=epochs, lr=lr)
        self.lbd = lbd

    def train(self, X, Y):
        m = X.shape[0] #n de amostras
        n = X.shape[1] #n de features
        self.history = {}
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            z = np.dot(X, self.theta)
            h = sigmoid(z)
            grad = np.dot(X.T, (h - Y)) / Y.size
            grad[1:] = grad[1:] + (self.lbd/m) * self.theta[1:]
            self.theta -= self.lr * grad
            self.history[epoch] = [self.theta[:], self.cost()]

    def cost(self, X=None, Y=None, theta=None):
        # x = X if X is not None else self.X
        # y = Y if Y is not None else self.Y
        # theta1 = theta if theta is not None else self.theta
        X = add_intersect(X) if X is not None else self.X
        Y = Y if Y is not None else self.Y
        theta = theta if theta is not None else self.theta

        m = X.shape[0]
        h = sigmoid(np.dot(X, theta))
        cost = (-Y * np.log(h) - (1 - Y) * np.log(1 - h))
        regul = np.dot(theta[1:], theta[1:]) * self.lbd / (2 * m)
        res = (np.sum(cost) / m) + regul
        return res