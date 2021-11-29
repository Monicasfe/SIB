from .model import Model
import numpy as np
##DT.predict(x) --> 0
##KNN.predict(x) --> 0          o fvote vai ser o valor com maior freq que neste caso seria o 0
##LogReg.predict(x) --> 1

def fvote(preds):
    """Conta a ocorrencia de cada valor e retorna qual e o max"""
    return max(set(preds), key=preds.count)

def majority(values):
    return max(set(values), key=values.count)

def average(values):
    return sum(values)/len(values)

class Ensemble(Model):

    def __init__(self, models_list, fvote, score):
        super().__init__()
        self.models = models_list
        self.fvote = fvote
        self.score = score

    def fit(self, dataset):
        self.dataset = dataset
        for model in self.models:
            model.fit(dataset)
        self.is_fitted = True

    def predict(self, x):
        assert self.is_fitted, "Model must be fitted before predicting"
        preds = [model.predict(x) for model in self.models]
        vote = self.fvote(preds)
        return vote

    def cost(self, X=None, Y=None):
        X = X if X is not None else self.dataset.X
        Y = Y if Y is not None else self.dataset.Y
        y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=X.T)
        return self.score(Y, y_pred)