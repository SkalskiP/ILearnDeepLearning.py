from src.base import Layer
import numpy as np


class ReluLayer(Layer):
    def __init__(self):
        self._Z = None

    def forward_pass(self, activation: np.array) -> np.array:
        self._Z = np.maximum(0, activation)
        return self._Z

    def backward_pass(self, activation: np.array) -> np.array:
        dZ = np.array(activation, copy=True)
        dZ[self._Z <= 0] = 0
        return dZ

    def update(self, lr: float) -> None:
        pass
