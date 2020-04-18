import numpy as np


def one_hot(y: np.array) -> np.array:
    y_ = np.zeros((y.size, y.max()+1))
    y_[np.arange(y.size), y] = 1
    return y_


def softmax(y: np.array) -> np.array:
    # Column wise softmax
    e_y = np.exp(y - np.max(y))
    return e_y / e_y.sum(axis=0)
