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

def accuracy_score(y_true, y_pred):
    """
    Classification performance metric
    """
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1

    accuracy = correct / len(y_true)
    return accuracy

def split_dataset_train_test(dataset, per_div = 0.8): #80 treino 20 teste
    size = dataset.X.shape[0]
    m = int(per_div*size) #m is train
    arr = np.arange(size)
    np.random.shuffle(arr)
    from ..data import Dataset
    train = Dataset(dataset.X[arr[:m]], dataset.Y[arr[:m]], dataset.xnames, dataset.yname)
    test = Dataset(dataset.X[arr[m:]], dataset.Y[arr[m:]], dataset.xnames, dataset.yname)
    return train, test

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse(y_true, y_pred, squared=True):
    """
    Mean square error regresion loss function
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errors = np.average((y_true - y_pred) ** 2, axis=0)
    if not squared:
        errors = np.sqrt(errors)
    return np.average(errors)

def add_intersect(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


def cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred)).sum()


def cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true


def r2_score(y_true, y_pred):
    """
    R^2 regression score function.
        R^2 = 1 - SS_res / SS_tot
    where SS_res is the residual sum of squares and SS_tot is the total
    sum of squares.
    :param numpy.array y_true : array-like of shape (n_samples,) Ground truth (correct) target values.
    :param numpy.array y_pred : array-like of shape (n_samples,) Estimated target values.
    :returns: score (float) R^2 score.
    """
    # Residual sum of squares.
    numerator = ((y_true - y_pred) ** 2).sum(axis=0)
    # Total sum of squares.
    denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0)
    # R^2.
    score = 1 - numerator / denominator
    return score

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def minibatch(X, batchsize=256, shuffle=True):
    N = X.shape[0]
    ix = np.arange(N)
    n_batches = int(np.ceil(N / batchsize))

    if shuffle:
        np.random.shuffle(ix)

    def mb_generator():
        for i in range(n_batches):
            yield ix[i * batchsize: (i + 1) * batchsize]

    return mb_generator(),