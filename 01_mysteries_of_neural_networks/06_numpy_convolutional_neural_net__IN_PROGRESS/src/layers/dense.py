from __future__ import annotations
from src.base import Layer
import numpy as np


class DenseLayer(Layer):
    def __init__(self, w: np.array, b: np.array):
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

    def update(self, lr: float) -> None:
        """
        :param lr -learning rate
        """
        self._w -= lr * self._dw
        self._b -= lr * self._db
