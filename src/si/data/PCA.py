import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler




class PCA:
    def __init__(self, n_components=2, method="svd"): #method  o metodo de calculo
        self.n_components = n_components
        available_methods = ["svd", "evd"]
        if method not in available_methods:
            raise Exception(f"Method not available. Please choose between: {available_methods}.")
        self.method = method

        def fit(self, dataset):
            pass

        def transform(self, dataset):
            x = dataset.X
            x_scaled = StandardScaler().fit_transform(x)  # normalizar os dados
            features = x_scaled.T #transposta dos dados normalizados
            if self.method == "svd":
                self.vecs, self.vals, rv = np.linalg.svd(features) #da vetores e valores, vetores + importantes
            else:
                cov_matrix = np.cov(features)
                self.vals, self.vecs = np.linalg.eig(cov_matrix)
            self.sorted_idx = np.argsort(self.vals)[::-1]  # indices ordenados por importancia das componentes, componete que explica mais vai para primeiro
            self.sorted_e_value = self.vals[self.sorted_idx]  # ordenar os valores pelos indices das colunas, valores so 1 dimensao
            self.sorted_e_vectors = self.vecs[:, self.sorted_idx]  # ordenar os vetores pelos indices das colunas, vetores 2 dimensoes
            if self.n_components > 0:
                if self.n_components > x.shape[1]:
                    warnings.warn("The number of components is larger than the number of features.")
                    self.n_components = x.shape[1]
                self.components_vector = self.sorted_e_vectors[:,
                                         0:self.n_components]  # vetores correspondentes ao numero de componentes selecionados
            else:
                warnings.warn("The number of components is lower than 0.")
                self.n_components = 1
                self.components_vector = self.sorted_e_vectors[:, 0:self.n_components]
            x_red = np.dot(self.components_vector.transpose(), features).transpose() #volta ao inicio como estava antes da trnasposta
            return x_red

        def fit_transform(self, dataset):
            x_red = self.transform(dataset)
            components_sum, components_values = self.explained_variances()
            return x_red, components_sum, components_values #vetores ordenados das componentes, a soma da varianacia e os resultados

        def explained_variances(self):
            self.components_values = self.sorted_e_value[0:self.n_components] / np.sum(self.sorted_e_value)#da os valores desde o primiero ate ao nº euq selecionakos de componentes
            return np.sum(self.components_values), self.components_values # soma dos compomentes e depois o valor de cada um dos componentes tipo 80 (60, 20)