from src.base import Layer
import numpy as np


class FlattenLayer(Layer):
    def __init__(self):
        self._shape = ()

    def forward_pass(self, a_prev: np.array) -> np.array:
        self._shape = a_prev.shape
        return np.ravel(a_prev).reshape(a_prev.shape[0], -1)

    def backward_pass(self, da_curr: np.array) -> np.array:
        return da_curr.reshape(self._shape)
