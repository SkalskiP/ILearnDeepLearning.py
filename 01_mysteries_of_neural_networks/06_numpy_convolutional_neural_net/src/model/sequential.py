from __future__ import annotations
from typing import List, Dict, Callable, Optional
import time

import numpy as np

from src.base import Layer, Optimizer
from src.utils.core import generate_batches, format_time
from src.utils.metrics import softmax_accuracy, softmax_cross_entropy


class SequentialModel:
    def __init__(self, layers: List[Layer], optimizer: Optimizer):
        self._layers = layers
        self._optimizer = optimizer

        self._train_acc = []
        self._test_acc = []
        self._train_loss = []
        self._test_loss = []

    def train(
        self,
        x_train: np.array,
        y_train: np.array,
        x_test: np.array,
        y_test: np.array,
        epochs: int,
        bs: int = 64,
        verbose: bool = False,
        callback: Optional[Callable[[SequentialModel], None]] = None
    ) -> None:
        """
        :param x_train - ND feature tensor with shape (n_train, ...)
        :param y_train - 2D one-hot labels tensor with shape (n_train, k)
        :param x_test - ND feature tensor with shape (n_test, ...)
        :param y_test - 2D one-hot labels tensor with shape (n_test, k)
        :param epochs - number of epochs used during model training
        :param bs - size of batch used during model training
        :param verbose - if set to True, model will produce logs during training
        :param callback - function that will be executed at the end of each epoch
        ------------------------------------------------------------------------
        n_train - number of examples in train data set
        n_test - number of examples in test data set
        k - number of classes
        """

        for epoch in range(epochs):
            epoch_start = time.time()
            y_hat = np.zeros_like(y_train)
            for idx, (x_batch, y_batch) in \
                    enumerate(generate_batches(x_train, y_train, bs)):

                y_hat_batch = self._forward(x_batch, training=True)
                activation = y_hat_batch - y_batch
                self._backward(activation)
                self._update()
                n_start = idx * bs
                n_end = n_start + y_hat_batch.shape[0]
                y_hat[n_start:n_end, :] = y_hat_batch

            self._train_acc.append(softmax_accuracy(y_hat, y_train))
            self._train_loss.append(softmax_cross_entropy(y_hat, y_train))

            y_hat = self._forward(x_test, training=False)
            test_acc = softmax_accuracy(y_hat, y_test)
            self._test_acc.append(test_acc)
            test_loss = softmax_cross_entropy(y_hat, y_test)
            self._test_loss.append(test_loss)

            if verbose:
                epoch_time = format_time(start_time=epoch_start, end_time=time.time())
                print("iter: {:05} | test loss: {:.5f} | test accuracy: {:.5f} | time: {}"
                      .format(epoch+1, test_loss, test_acc, epoch_time))

    def predict(self, x: np.array) -> np.array:
        """
        :param x - ND feature tensor with shape (n, ...)
        :output 2D one-hot labels tensor with shape (n, k)
        ------------------------------------------------------------------------
        n - number of examples in data set
        k - number of classes
        """
        return self._forward(x, training=False)

    @property
    def history(self) -> Dict[str, List[float]]:
        return {
            "train_acc": self._train_acc,
            "test_acc": self._test_acc,
            "train_loss": self._train_loss,
            "test_loss": self._test_loss
        }

    def _forward(self, x: np.array, training: bool) -> np.array:
        activation = x
        for idx, layer in enumerate(self._layers):
            activation = layer.forward_pass(a_prev=activation, training=training)
        return activation

    def _backward(self, x: np.array) -> None:
        activation = x
        for layer in reversed(self._layers):
            activation = layer.backward_pass(da_curr=activation)

    def _update(self) -> None:
        self._optimizer.update(layers=self._layers)


