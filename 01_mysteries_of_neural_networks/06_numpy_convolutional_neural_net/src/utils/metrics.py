import numpy as np

from src.utils.core import convert_prob2one_hot


def calculate_accuracy(y_hat: np.array, y: np.array) -> float:
    """
    k - number of classes
    N - number of items
    :param y_hat - softmax output array with (N, k) shape
    :param y - one hot ground truth array with (N, k) shape
    """
    y_hat = convert_prob2one_hot(y_hat)
    return (y_hat == y).all(axis=1).mean()


def multi_class_cross_entropy_loss(Y_hat, Y, eps=1e-12) -> float:
    m = Y_hat.shape[0]
    Y_hat = np.clip(Y_hat, eps, 1. - eps)
    loss = - np.sum(Y * np.log(Y_hat)) / m
    return loss