from __future__ import annotations

from typing import Tuple

from src.base import Layer
import numpy as np


class ConvLayer2D(Layer):
    def __init__(self, W: np.array, b: np.array):
        self._W, self._b = W, b

    @classmethod
    def initialize(
        cls,
        filters: int,
        kernel_shape: Tuple[int, int, int]
    ) -> ConvLayer2D:
        W = np.random.randn(filters, *kernel_shape) * 0.1
        b = np.random.randn(filters) * 0.1
        return cls(W=W, b=b)

    def forward_pass(self, activation: np.array) -> np.array:
        pass

    def backward_pass(self, activation: np.array) -> np.array:
        pass

    def update(self, lr: float) -> None:
        pass

    @staticmethod
    def single_convolution_step(
        activation: np.array,
        filter_W: np.array,
        filter_b: float
    ) -> np.array:
        """
        f - filter dimension
        c - number of channels
        :param activation - slice of layer activation array with (f, f, c) shape
        :param filter_W - single filter weights array with (f, f, c) shape
        :param filter_b - single filter bias
        """
        return np.sum(np.multiply(activation, filter_W)) + filter_b
