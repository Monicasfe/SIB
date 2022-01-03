import itertools

import numpy as np
from src.si.util.util import split_dataset_train_test

class CrossValidationScore():

    def __init__(self, model, dataset, score=None, **kwargs):
        self.model = model
        self.dataset = dataset
        self.score = score
        self.cv = kwargs.get("cv", 3)
        self.split = kwargs.get("split", 0.8)
        self.train_scores = None #funcao da semana passada de train_test split
        self.test_scores = None
        self.ds = None

    def run(self):
        train_scores = [] #scores dos train
        test_scores = [] #scores dos test
        ds = [] #guarda os datasets, ou seja os resultados dos splits
        pred_y = []
        true_y = []
        for _ in range(self.cv): #o _ serve porque no precisamos de chamar o valor da iteracao por isso nao vale a pena guardar
            train, test = split_dataset_train_test(self.dataset, self.split)
            ds.append((train, test))
            self.model.fit(train)
            if not self.score:
                train_scores.append(self.model.cost())
                test_scores.append(self.model.cost(test.X, test.Y))
                pred_y.append(y_test)
                true_y.append(test.Y)
            else:
                y_train = np.ma.apply_along_axis(self.model.predict, axis=0, arr=train.X.T)
                train_scores.append(self.score(train.Y, y_train))
                y_test = np.ma.apply_along_axis(self.model.predict, axis=0, arr=test.X.T)
                test_scores.append(self.score(test.Y, y_test))
                pred_y.append(y_test)
                true_y.append(test.Y)
        self.train_scores = train_scores
        self.test_scores = test_scores
        self.ds = ds
        self.pred_y = pred_y
        self.true_y = true_y
        return train_scores, test_scores

    def to_dataframe(self):
        import pandas as pd
        assert self.train_scores and self.test_scores, "Need to run the run function first"
        return pd.DataFrame({"Train Sores": self.train_scores, "Test scores": self.test_scores})

class GridSearchCV():

    def __init__(self, model, dataset, parameters, **kwargs):
        """
        :param model: modelo em causa
        :param dataset: o dataset escolhido
        """
        self.model = model
        self.dataset = dataset
        #verifica se os parametros estao corretos ou seja vai a lista ver ser se os atributos existem na lista ou nao
        hasparam = [hasattr(self.model, param) for param in parameters]
        if np.all(hasparam):
            self.parameters = parameters
        else:
            index = hasparam.index(False)
            keys = list(parameters.keys())
            raise ValueError(f"Wrong parameters: {keys[index]}")
        self.kwargs = kwargs


    def run(self):
        self.results = []
        attrs = list(self.parameters.keys())
        values = list(self.parameters.values())
        for conf in itertools.product(*values): #produto carteziano dos valores
            for i in range(len(attrs)):
                setattr(self.model, attrs[i], conf[i]) #da um atributo a um objeto e o seu valor attrs = aatributo conf = valores
            scores = CrossValidationScore(self.model, self.dataset, **self.kwargs).run()
            self.results.append((conf, scores))
        return self.results


    def to_dataframe(self):
        import pandas as pd
        assert self.results, "The grid search needs to be ran"
        data = {}
        for i, k in enumerate(self.parameters.keys()):
            v = []
            for r in self.results:
                v.append(r[0][i])
            data[k] = v
        for i in range(len(self.results[0][1][0])):
            train = []
            test = []
            for r in self.results:
                train.append(r[1][0][i])
                test.append(r[1][1][i])
                data[f"Train {str(i + 1)} Scores"] = train
                data[f"Test {str(i + 1)} Scores"] = test

        return pd.DataFrame.from_dict(data=data)
