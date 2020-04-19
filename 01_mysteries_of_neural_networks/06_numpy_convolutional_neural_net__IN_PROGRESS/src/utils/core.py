import numpy as np


def convert_categorical2one_hot(y: np.array) -> np.array:
    """
    k - number of classes
    N - number of instances
    :param y - categorical array with (N, 1) shape
    :return one hot array with (N, k) shape
    """
    one_hot_matrix = np.zeros((y.size, y.max() + 1))
    one_hot_matrix[np.arange(y.size), y] = 1
    return one_hot_matrix


def convert_prob2one_hot(probs: np.array) -> np.array:
    """
    k - number of classes
    N - number of instances
    :param probs - softmax output array with (k, N) shape
    :return one hot array with (k, N) shape
    """
    class_idx = np.argmax(probs, axis=0)
    one_hot_matrix = np.zeros(probs.shape)
    one_hot_matrix[class_idx, np.arange(probs.shape[1])] = 1
    return one_hot_matrix


def softmax(y: np.array) -> np.array:
    # Column wise softmax
    e_y = np.exp(y - np.max(y))
    return e_y / e_y.sum(axis=0)
