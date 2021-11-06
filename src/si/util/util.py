import itertools
import numpy as np

# Y is reserved to idenfify dependent variables

ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXZ'


__all__ = ['label_gen']#, 'summary']

def label_gen(n):
    """ Generates a list of n distinct labels similar to Excel"""
    def _iter_all_strings():
        size = 1
        while True:
            for s in itertools.product(ALPHA, repeat=size):
                yield "".join(s)
            size += 1

    generator = _iter_all_strings()

    def gen():
        for s in generator:
            return s

    return [gen() for _ in range(n)]

def euclidean(x, y):
    """
    Computes the difference o a n dimensional vetor x and a list of m and summing
    xhape = n     """
    #dist = (np.absolute(x - y))
    #dist = ((x - y) ** 2).sum(axis=1)
    dist = (x - y) ** 2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    return dist


def manhattan(x, y):
    dist = np.abs(x - y)
    dist = np.sum(dist)
    return dist
