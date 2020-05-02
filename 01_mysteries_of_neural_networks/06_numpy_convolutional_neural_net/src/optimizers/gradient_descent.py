from typing import List

from src.base import Optimizer, Layer


class GradientDescent(Optimizer):
    def __init__(self, lr: float):
        """
        :param lr - learning rate
        """
        self._lr = lr

    def update(self, layers: List[Layer]) -> None:
        for layer in layers:
            weights, gradients = layer.weights, layer.gradients
            if weights is None or gradients is None:
                continue

            (w, b), (dw, db) = weights, gradients
            layer.set_wights(
                w = w - self._lr * dw,
                b = b - self._lr * db
            )