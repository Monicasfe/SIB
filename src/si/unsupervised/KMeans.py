import numpy as np
from src.si.util.util import euclidean, manhattan


class KMeans:

    def __init__(self, k: int, distance, max_iterations = 500):
        dists = ["euclidean", "manhattan"]
        self.K = k
        self.centroids = None
        self.max_iter = max_iterations

        if distance not in dists:
            raise Exception(f"Distance is no available, those are the ones that can be used: {dists}")
        elif distance == "euclidean":
            self.distance = euclidean
        else:
            self.distance = manhattan


    def fit(self, dataset):
        x = dataset.X
        self.min = np.min(x, axis=0)
        self.max = np.max(x, axis=0)



    def init_centroids(self, dataset):
        x = dataset.X
        self.centroids = np.array([np.random.uniform(low=self.min[i],
                                                     high=self.max[i],
                                                     size=(self.K,)) for i in range(x.shape[1])]).T


    def get_closest_centroid(self, x):
        dist = self.distance(x, self.centroids)
        closest_centroid_index = np.argmin(dist, axis=0)
        return closest_centroid_index

    def transform(self, dataset):
        self.init_centroids(dataset)
        print(self.centroids)
        x = dataset.X
        changed = False
        count = 0
        old_idxs = np.zeros(x.shape[0])
        while count < self.max_iter and changed==False:
            #array of indexes of nearest centroid
            idxs = np.apply_along_axis(self.get_closest_centroid, axis=0, arr=x.T)
            centroid = []
            for i in range(self.K):
                centroid.append(np.mean(x[idxs == i], axis=0))
            #cent = [np.mean(X[idxs == i], axis=0) for i in range(self.K)]
            self.centroids = np.array(centroid)
            changed = np.all(old_idxs == idxs) #muda o changed para false
            old_idxs = idxs
            count += 1
        return self.centroids, old_idxs

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)
