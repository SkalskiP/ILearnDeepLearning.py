from typing import List
import numpy as np

from src.base import Layer
from src.utils.metrics import get_accuracy_value


class Sequential:
    def __init__(self, layers: List[Layer]):
        self._layers = layers

    def forward(self, input: np.array) -> np.array:
        activation = input
        for layer in self._layers:
            activation = layer.forward_pass(activation=activation)
        return activation

    def backward(self, input: np.array) -> np.array:
        activation = input
        for layer in self._layers:
            activation = layer.backward_pass(activation=activation)
        return activation

    def update(self, lr: float) -> None:
        for layer in self._layers:
            layer.update(lr=lr)

    def train(self, X: np.array, y: np.array, epochs: int, lr: float) -> np.array:
        for epoch in range(epochs):
            y_hat = self.forward(X)
            activation = - (np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))
            self.backward(activation)
            self.update(lr=lr)

            accuracy = get_accuracy_value(y_hat, y)
            print(accuracy)

    def predict(self, X: np.array) -> np.array:
        return self.forward(X)
