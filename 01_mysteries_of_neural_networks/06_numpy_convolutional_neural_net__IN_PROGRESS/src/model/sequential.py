from typing import List
import numpy as np

from src.base import Layer
from src.utils.core import generate_batches
from src.utils.metrics import calculate_accuracy, multi_class_cross_entropy_loss


class SequentialModel:
    def __init__(self, layers: List[Layer]):
        self._layers = layers

    def forward(self, input: np.array) -> np.array:
        activation = input
        for layer in self._layers:
            activation = layer.forward_pass(a_prev=activation)
        return activation

    def backward(self, input: np.array) -> None:
        activation = input
        for layer in reversed(self._layers):
            activation = layer.backward_pass(da_curr=activation)

    def update(self, lr: float) -> None:
        for layer in self._layers:
            layer.update(lr=lr)

    def train(
        self,
        x_train: np.array,
        y_train: np.array,
        x_test: np.array,
        y_test: np.array,
        epochs: int,
        lr: float,
        batch_size: int = 64,
        test_frequency: int = 10
    ) -> np.array:

        for epoch in range(epochs):
            for batch_idx, (X_batch, y_batch) in enumerate(generate_batches(x_train, y_train, batch_size)):
                y_hat = self.forward(X_batch)
                activation = y_hat - y_batch
                self.backward(activation)
                self.update(lr=lr)

            if (epoch + 1) % test_frequency == 0:
                y_hat = self.forward(x_test)
                accuracy = calculate_accuracy(y_hat, y_test)
                loss = multi_class_cross_entropy_loss(y_hat, y_test)
                print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}"
                      .format(epoch+1, loss, accuracy))

    def predict(self, x: np.array) -> np.array:
        return self.forward(x)
