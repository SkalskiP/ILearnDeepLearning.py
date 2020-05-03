import numpy as np

from src.base import Layer


class FlattenLayer(Layer):

    def __init__(self):
        self._shape = ()

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        """
        :param a_prev - ND tensor with shape (n, ..., channels)
        :output - 1D tensor with shape (n, 1)
        ------------------------------------------------------------------------
        n - number of examples in batch
        """
        self._shape = a_prev.shape
        return np.ravel(a_prev).reshape(a_prev.shape[0], -1)

    def backward_pass(self, da_curr: np.array) -> np.array:
        """
        :param da_curr - 1D tensor with shape (n, 1)
        :output - ND tensor with shape (n, ..., channels)
        ------------------------------------------------------------------------
        n - number of examples in batch
        """
        return da_curr.reshape(self._shape)
