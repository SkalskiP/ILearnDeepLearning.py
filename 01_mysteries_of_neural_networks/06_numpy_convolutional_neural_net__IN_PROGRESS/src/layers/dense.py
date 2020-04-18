from __future__ import annotations
from src.base import Layer
import numpy as np


class DenseLayer(Layer):
    def __init__(self, W: np.array, b: np.array):
        self._W, self._b = W, b
        self._dW, self._db = None, None

        self._Z, self._A = None, None

    @classmethod
    def initialize(cls, input_dim: int, output_dim: int) -> DenseLayer:
        W = np.random.randn(output_dim, input_dim) * 0.1
        b = np.random.randn(output_dim, 1) * 0.1
        return cls(W=W, b=b)

    def forward_pass(self, activation: np.array) -> np.array:
        self._Z = np.dot(self._W, activation) + self._b
        self._A = np.array(activation, copy=True)
        return self._Z

    def backward_pass(self, activation: np.array) -> np.array:
        m = self._A.shape[1]
        self._dW = np.dot(activation, self._A.T) / m
        self._db = np.sum(activation, axis=1, keepdims=True) / m
        return np.dot(self._W.T, activation)

    def update(self, lr: float) -> None:
        self._W -= lr * self._dW
        self._b -= lr * self._db
