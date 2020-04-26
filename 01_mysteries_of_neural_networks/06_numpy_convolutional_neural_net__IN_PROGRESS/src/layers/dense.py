from __future__ import annotations

from typing import Optional, Tuple

from src.base import Layer
import numpy as np


class DenseLayer(Layer):
    def __init__(self, w: np.array, b: np.array):
        """
        :param w - 2D weights tensor with shape (units_curr, units_prev)
        :param b - 1D bias tensor with shape (1, units_curr)
        ------------------------------------------------------------------------
        units_prev - number of units in previous layer
        units_curr -  number of units in current layer
        """
        self._w, self._b = w, b
        self._dw, self._db = None, None
        self._a_prev = None

    @classmethod
    def initialize(cls, units_prev: int, units_curr: int) -> DenseLayer:
        """
        :param units_prev - positive integer, number of units in previous layer
        :param units_curr - positive integer, number of units in current layer
        """
        w = np.random.randn(units_curr, units_prev) * 0.1
        b = np.random.randn(1, units_curr) * 0.1
        return cls(w=w, b=b)

    @property
    def weights(self) -> Optional[Tuple[np.array, np.array]]:
        return self._w, self._b

    @property
    def gradients(self) -> Optional[Tuple[np.array, np.array]]:
        if self._dw is None or self._db is None:
            return None
        return self._dw, self._db

    def forward_pass(self, a_prev: np.array) -> np.array:
        """
        :param a_prev - 2D tensor with shape (n, units_prev)
        :output - 2D tensor with shape (n, units_curr)
        ------------------------------------------------------------------------
        n - number of examples in batch
        units_prev - number of units in previous layer
        units_curr -  number of units in current layer
        """
        self._a_prev = np.array(a_prev, copy=True)
        return np.dot(a_prev, self._w.T) + self._b

    def backward_pass(self, da_curr: np.array) -> np.array:
        """
        :param da_curr - 2D tensor with shape (n, units_curr)
        :output - 2D tensor with shape (n, units_prev)
        ------------------------------------------------------------------------
        n - number of examples in batch
        units_prev - number of units in previous layer
        units_curr -  number of units in current layer
        """
        n = self._a_prev.shape[0]
        self._dw = np.dot(da_curr.T, self._a_prev) / n
        self._db = np.sum(da_curr, axis=0, keepdims=True) / n
        return np.dot(da_curr, self._w)

    def set_wights(self, w: np.array, b: np.array) -> None:
        """
        :param w - 2D weights tensor with shape (units_curr, units_prev)
        :param b - 1D bias tensor with shape (1, units_curr)
        ------------------------------------------------------------------------
        units_prev - number of units in previous layer
        units_curr -  number of units in current layer
        """
        self._w = w
        self._b = b
