import numpy as np
from .model import Model
from src.si.util.util import accuracy_score

class Naive_Bayes(Model):

    def __init__(self):
        """
                Abstract class defining an interface for supervised learning models.
                """
        super(Naive_Bayes).__init__()
        self.is_fitted = False

    def fit(self, dataset):
        """
        P(A|B) = (P(B|A) * P(A)) / P(B)
        posterior = likelihood * prior / normalizing constant
        Where:
            A = Class
            B = Data
            P(A|B): Posterior (is the posterior probability of class (target) given predictor (attribute))
            P(B|A): Likelihood (is the likelihood which is the probability of predictor given class.)
            P(A): Prior (is the prior probability of predictor.)
            P(B) = Normalizing constant or evidence (is the prior probability of predictor.)
        """
        self.dataset = dataset
        X, Y = self.dataset.getXy()
        n_rows = X.shape[0]
        self.classes = np.unique(Y)
        X_byclass = np.array([X[Y == target] for target in self.classes])
        self.prior = np.array([len(X_Class) / n_rows for X_Class in X_byclass])

        self.mean = np.asarray([np.mean(target, axis=0) for target in X_byclass])
        self.var = np.asarray([np.var(target, axis=0) for target in X_byclass])

        self.is_fitted = True

    def gaussian_NB(self, x ):
        exponent = np.exp(-((x - self.mean) ** 2 / (2 * self.var ** 2)))
        proba = (1 / (np.sqrt(2 * np.pi) * self.var)) * exponent #self.var sem ser **2 pooprque esta fora do sqrt
        return proba

    def predict(self, x):
        assert self.is_fitted, "Model must be fitted before proceed"
        probs = np.prod(self.gaussian_NB(x), axis=1) * self.prior
        idx = np.argmax(probs)
        prediction = self.classes[idx]
        return prediction

    def cost(self):
        y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=self.dataset.X.T)
        return accuracy_score(self.dataset.Y, y_pred)
