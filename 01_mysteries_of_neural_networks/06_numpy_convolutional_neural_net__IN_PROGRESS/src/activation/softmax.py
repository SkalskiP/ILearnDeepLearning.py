from src.base import Layer
import numpy as np


class SoftmaxLayer(Layer):
    def __init__(self):
        self._Z = None

    def forward_pass(self, activation: np.array) -> np.array:
        e = np.exp(activation - activation.max())
        self._Z = e / np.sum(e, axis=0, keepdims=True)
        return self._Z

    def backward_pass(self, activation: np.array) -> np.array:
        return activation

    def update(self, lr: float) -> None:
        pass
