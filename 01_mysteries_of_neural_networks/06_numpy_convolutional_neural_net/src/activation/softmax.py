from src.base import Layer
import numpy as np


class SoftmaxLayer(Layer):
    def __init__(self):
        self._z = None

    def forward_pass(self, a_prev: np.array) -> np.array:
        """
        :param a_prev - 2D tensor with shape (n, k)
        :output 2D tensor with shape (n, k)
        ------------------------------------------------------------------------
        n - number of examples in batch
        k - number of classes
        """
        e = np.exp(a_prev - a_prev.max(axis=1, keepdims=True))
        self._z = e / np.sum(e, axis=1, keepdims=True)
        return self._z

    def backward_pass(self, da_curr: np.array) -> np.array:
        """
        :param da_curr - 2D tensor with shape (n, k)
        :output 2D tensor with shape (n, k)
        ------------------------------------------------------------------------
        n - number of examples in batch
        k - number of classes
        """
        return da_curr

