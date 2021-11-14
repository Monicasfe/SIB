from src.si.util.util import euclidean, accuracy_score
from .model import Model
import numpy as np

class KNN(Model):

    def __init__(self, n_neighbors, classification=True):
        """
                Abstract class defining an interface for supervised learning models.
                """
        super(KNN).__init__()
        self.n_neighbors = n_neighbors
        self.is_fitted = False
        self.classification = classification

    def fit(self, dataset):
        self.dataset = dataset
        self.is_fitted = True

    def get_neighbors(self, x):
        dist = euclidean(x, self.dataset.X)
        ord_idx = np.argsort(dist)
        return ord_idx[:self.n_neighbors]

    def predict(self, x):
        assert self.is_fitted, "Model must be fitted before prediction"
        neighbors = self.get_neighbors(x)
        values = self.dataset.Y[neighbors].tolist()

        #prediction = max(set(values), key=values.count)  # o maximo que aparece, ou seja o que aparece mais vezes

        #Esta implementaçao nao e mesmo necessaria e amis para funcuionar com regressao tambem
        if self.classification:
            prediction = max(set(values), key=values.count) #o maximo que aparece, ou seja o que aparece mais vezes, com labels

        else:
            prediction = sum(values)/len(values) #para funcionar com regressao valores

        return prediction


    def cost(self):
        y_pred = np.ma.apply_along_axis(self.predict, #o ma aqui indica que e com mascara
                                        axis=0, arr=self.dataset.X.T)
        return accuracy_score(self.dataset.Y, y_pred)


