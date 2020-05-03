import numpy as np

from src.base import Layer


class ReluLayer(Layer):
    def __init__(self):
        self._z = None

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        """
        :param a_prev - ND tensor with shape (n, ..., channels)
        :output ND tensor with shape (n, ..., channels)
        ------------------------------------------------------------------------
        n - number of examples in batch
        """
        self._z = np.maximum(0, a_prev)
        return self._z

    def backward_pass(self, da_curr: np.array) -> np.array:
        """
        :param da_curr - ND tensor with shape (n, ..., channels)
        :output ND tensor with shape (n, ..., channels)
        ------------------------------------------------------------------------
        n - number of examples in batch
        """
        dz = np.array(da_curr, copy=True)
        dz[self._z <= 0] = 0
        return dz

