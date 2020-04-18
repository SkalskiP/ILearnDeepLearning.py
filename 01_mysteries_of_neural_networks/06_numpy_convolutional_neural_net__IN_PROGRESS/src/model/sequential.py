from typing import List
import numpy as np

from src.base import Layer
from src.utils.core import softmax
from src.utils.metrics import get_accuracy_value


class SequentialModel:
    def __init__(self, layers: List[Layer]):
        self._layers = layers

    def forward(self, input: np.array) -> np.array:
        activation = input
        for layer in self._layers:
            activation = layer.forward_pass(activation=activation)
        return activation

    def backward(self, input: np.array) -> None:
        activation = input
        for layer in reversed(self._layers):
            activation = layer.backward_pass(activation=activation)

    def update(self, lr: float) -> None:
        for layer in self._layers:
            layer.update(lr=lr)

    def train(self, X: np.array, y: np.array, epochs: int, lr: float) -> np.array:
        print(y.shape)
        for epoch in range(epochs):
            y_hat = self.forward(X)

            # eps = 0.000000001
            # activation = - (np.divide(y, y_hat + eps) - np.divide(1 - y, 1 - y_hat + eps))

            # activation = y_hat - y

            # multi class cross entropy
            # epsilon=1e-12
            # y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
            activation = softmax(y_hat) - y

            self.backward(activation)
            self.update(lr=lr)

            accuracy = get_accuracy_value(y_hat, y)
            print("accuracy:", accuracy)

    def predict(self, X: np.array) -> np.array:
        return self.forward(X)
