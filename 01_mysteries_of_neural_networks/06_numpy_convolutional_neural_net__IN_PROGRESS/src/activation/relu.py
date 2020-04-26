from src.base import Layer
import numpy as np


class ReluLayer(Layer):
    def __init__(self):
        self._z = None

    def forward_pass(self, a_prev: np.array) -> np.array:
        self._z = np.maximum(0, a_prev)
        return self._z

    def backward_pass(self, da_curr: np.array) -> np.array:
        dz = np.array(da_curr, copy=True)
        dz[self._z <= 0] = 0
        return dz

    def update(self, lr: float) -> None:
        pass
