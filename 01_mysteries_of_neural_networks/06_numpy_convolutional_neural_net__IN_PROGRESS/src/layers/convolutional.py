from __future__ import annotations

from typing import Tuple

from src.base import Layer
import numpy as np

from src.errors import InvalidPaddingModeError


class ConvLayer2D(Layer):
    def __init__(self, W: np.array, b: np.array, padding: str = 'valid'):
        """
        f - filter count
        fd - filter dimension
        c - number of channels
        :param W - filter weights array with (f, fd, fd, c) shape
        :param b - filter bias array with (f, ) shape
        :param padding - flag describing type of activation padding valid/same
        """
        self._W, self._b = W, b
        self._padding = padding
        self._dW, self._db = None, None

    @classmethod
    def initialize(
        cls,
        filters: int,
        kernel_shape: Tuple[int, int, int],
        padding: str = 'valid'
    ) -> ConvLayer2D:
        W = np.random.randn(filters, *kernel_shape) * 0.1
        b = np.random.randn(filters) * 0.1
        return cls(W=W, b=b, padding=padding)

    def forward_pass(self, activation: np.array) -> np.array:
        pass

    def backward_pass(self, activation: np.array) -> np.array:
        pass

    def update(self, lr: float) -> None:
        self._W -= lr * self._dW
        self._b -= lr * self._db

    @staticmethod
    def pad_activation(
        activation: np.array,
        fd: int,
        mode: str
    ) -> np.array:
        """
        N - number of instances
        ad - activation dimension
        c - number of channels
        :param activation - activation array with (ad, ad, c, N) shape
        :param fd - filter dimension
        :param mode - flag describing type of activation padding valid/same
        for example padding in same mode with width equal to 2 will change
        activation shape from (11, 11, 9, 64) to (15, 15, 9, 64)
        """
        if mode == 'valid':
            return activation
        elif mode == 'same':
            pad = fd // 2
            return np.pad(
                array=activation,
                pad_width=((pad, pad), (pad, pad), (0, 0), (0, 0)),
                mode='constant'
            )
        else:
            raise InvalidPaddingModeError(
                f"The padding value can only be equal to valid or same, "
                f"got {mode} instead.")

    @staticmethod
    def single_convolution_step(
        activation: np.array,
        filter_W: np.array,
        filter_b: float
    ) -> np.array:
        """
        fd - filter dimension
        c - number of channels
        :param activation - slice of layer activation array with (fd, fd, c) shape
        :param filter_W - single filter weights array with (fd, fd, c) shape
        :param filter_b - single filter bias
        """
        return np.sum(np.multiply(activation, filter_W)) + filter_b
