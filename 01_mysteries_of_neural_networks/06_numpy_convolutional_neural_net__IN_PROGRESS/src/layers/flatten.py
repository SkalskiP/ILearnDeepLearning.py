from src.base import Layer
import numpy as np


class FlattenLayer(Layer):
    def __init__(self):
        self._shape = ()

    def forward_pass(self, activation: np.array) -> np.array:
        # print(f"FlattenLayer - forward_pass - activation shape: {activation}")
        self._shape = activation.shape
        return np.ravel(activation).reshape(-1, activation.shape[-1])

    def backward_pass(self, activation: np.array) -> np.array:
        # print(f"FlattenLayer - backward_pass - activation shape: {activation}")
        return activation.reshape(self._shape)

    def update(self, lr: float) -> None:
        pass
