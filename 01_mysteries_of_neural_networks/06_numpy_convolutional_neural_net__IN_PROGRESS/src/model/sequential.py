from typing import List, Tuple
import numpy as np

from src.base import Layer
from src.utils.core import softmax, generate_batches
from src.utils.metrics import calculate_accuracy, multi_class_cross_entropy_loss


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

    def _get_batch(self, X: np.array, y: np.array, epoch: int, batch_size=64
                   ) -> Tuple[np.array, np.array]:
        batch_count = X.shape[1] // batch_size
        batch_idx = epoch % batch_count
        X_batch = X[:, batch_idx * batch_size: (batch_idx + 1) * batch_size]
        y_batch = y[:, batch_idx * batch_size: (batch_idx + 1) * batch_size]
        return X_batch, y_batch

    def train(
        self,
        x_train: np.array,
        y_train: np.array,
        x_test: np.array,
        y_test: np.array,
        epochs: int,
        lr: float,
        batch_size=64
    ) -> np.array:

        for epoch in range(epochs + 1):
            for X_batch, y_batch in generate_batches(x_train, y_train, batch_size):
                y_hat = self.forward(X_batch)
                activation = softmax(y_hat) - y_batch
                self.backward(activation)
                self.update(lr=lr)

            if epoch % 10 == 0:
                y_hat = self.forward(x_test)
                accuracy = calculate_accuracy(y_hat, y_test)
                loss = multi_class_cross_entropy_loss(y_hat, y_test)
                print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}"
                      .format(epoch, loss, accuracy))

    def predict(self, x: np.array) -> np.array:
        return self.forward(x)
