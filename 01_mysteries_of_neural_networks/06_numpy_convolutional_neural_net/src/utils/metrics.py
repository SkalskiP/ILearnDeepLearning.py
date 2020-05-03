import numpy as np

from src.utils.core import convert_prob2one_hot


def softmax_accuracy(y_hat: np.array, y: np.array) -> float:
    """
    :param y_hat - 2D one-hot prediction tensor with shape (n, k)
    :param y - 2D one-hot ground truth labels tensor with shape (n, k)
    ----------------------------------------------------------------------------
    n - number of examples in batch
    k - number of classes
    """
    y_hat = convert_prob2one_hot(y_hat)
    return (y_hat == y).all(axis=1).mean()


def softmax_cross_entropy(y_hat, y, eps=1e-20) -> float:
    """
    :param y_hat - 2D one-hot prediction tensor with shape (n, k)
    :param y - 2D one-hot ground truth labels tensor with shape (n, k)
    ----------------------------------------------------------------------------
    n - number of examples in batch
    k - number of classes
    """
    n = y_hat.shape[0]
    return - np.sum(y * np.log(np.clip(y_hat, eps, 1.))) / n
