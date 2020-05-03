from src.base import Layer

import numpy as np


class DropoutLayer(Layer):

    def __init__(self, keep_prob):
        """
        :param keep_prob - probability that given unit will not be dropped out
        """
        self._keep_prob = keep_prob
        self._mask = None

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        if training:
            self._mask = (np.random.rand(*a_prev.shape) < self._keep_prob)
            return self._apply_mask(a_prev, self._mask)
        else:
            return a_prev

    def backward_pass(self, da_curr: np.array) -> np.array:
        return self._apply_mask(da_curr, self._mask)

    def _apply_mask(self, array: np.array, mask: np.array) -> np.array:
        array *= mask
        array /= self._keep_prob
        return array
