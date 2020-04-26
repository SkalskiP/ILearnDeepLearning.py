from src.base import Layer
import numpy as np


class SoftmaxLayer(Layer):
    def __init__(self):
        self._z = None

    def forward_pass(self, a_prev: np.array) -> np.array:
        e = np.exp(a_prev - a_prev.max())
        self._z = e / np.sum(e, axis=1, keepdims=True)
        return self._z

    def backward_pass(self, da_curr: np.array) -> np.array:
        return da_curr

    def update(self, lr: float) -> None:
        pass
