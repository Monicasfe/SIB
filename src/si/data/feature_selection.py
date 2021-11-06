import numpy as np
from scipy import stats
from scipy.stats import f_oneway
from copy import copy
import warnings

from src.si.data import Dataset

class VarianceThreshold:

    def __init__(self, threshold):
        if threshold < 0:
            warnings.warn("The Treshold mut be a non-negative value.")
        self.threshold = threshold

    def fit(self, dataset):
        X = dataset.X
        self._var = np.var(X, axis=0) #axis=0 sao as colunas

    def transform(self, dataset, inline=False):
        X = dataset.X
        cond = self._var > self.threshold #isto  1 array de boelanos
        idxs = [i for i in range(len(cond)) if cond[i]]
        X_trans = X[:, idxs]
        xnames = [dataset.xnames[i] for i in idxs]
        if inline:
            dataset.X = X_trans
            dataset.xnames = xnames
            return dataset
        else:
            #from .dataset import Dataset
            return Dataset(copy(X_trans), copy(dataset.Y), xnames, copy(dataset.yname))

    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline=inline)

class selectKBest:
    """
    Features reduction
    """

    def __init__(self, K, score_function="f_regression"):
        available_sf = ["f_classif", "f_regression"]

        if score_function not in available_sf: #verify if function is available
            raise Exception(f"Scoring function not available. Please choose between : {available_sf}.")
        elif score_function == "f_classif":
            self.function = f_classif
        else:
            self.function = f_regression
        if K <= 0:
            raise Exception("The K value must be higher than 0.")
        else:
            self.k = K

    def fit(self, dataset):
        self.F_stat, self.pvalue = self.function(dataset)

    def transform(self, dataset, inline=False):
        X, Xnames = dataset.X, dataset.xnames

        if self.k > X.shape[1]:
            warnings.wanr("The K value provided is greater than the number of features. "
                          "All features wil be selected")
            self.K = int(X.shape[1])
        sel_feats = np.argsort(self.F_stat)[-self.k:] #ordem de valores F_stats
        sel_feats = sorted(sel_feats) #ordem das labels
        x = X[:, sel_feats]
        x_names = [Xnames[index] for index in sel_feats]

        if inline:
            dataset.X = x
            dataset.xnames = x_names
        else:
            return Dataset(x, copy(dataset.Y), x_names, copy(dataset.yname))

    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline=inline)


def f_classif(dataset):
    X, y = dataset.getXy()
    args = []
    for k in np.unique(y):
        args.append(X[y == k, :])
    F_stat, pvalue = f_oneway(*args)
    return F_stat, pvalue

def f_regression(dataset):
    X, y = dataset.getXy()
    cor_coef = np.array([stats.pearsonr(X[:, 1], y)[0] for i in range(X.shape[1])])
    dof = y.size - 2 #degrees of freedom
    cor_coef_sqrd = cor_coef ** 2
    F = cor_coef_sqrd / (1 - cor_coef_sqrd) * dof
    p = stats.f.sf(F, 1, dof)
    return F, p

