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
        self._A = None

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
        """
        N - number of items
        ad - activation dimension
        c - number of channels
        :param activation - activation array with (ad, ad, c, N) shape
        """
        self._A = np.array(activation, copy=True)

        _, fd, _, _ = self._W.shape

        output_shape = ConvLayer2D.calculate_forward_pass_output_shape(
            activation=activation,
            W=self._W,
            mode=self._padding
        )

        Z = np.zeros(shape=output_shape)

        activation = ConvLayer2D.pad_activation(
            activation=activation,
            fd=fd,
            mode=self._padding
        )

        height, width, channels, items = output_shape
        for i in range(items):
            for h in range(height):
                for w in range(width):
                    for c in range(channels):
                        filter_W = self._W[c, :, :, :]
                        filter_b = self._b[c]
                        activation_slice = activation[h:h+fd, w:w+fd, :, i]
                        Z[h, w, c, i] = ConvLayer2D.single_convolution_step(
                            activation_slice=activation_slice,
                            filter_W=filter_W,
                            filter_b=filter_b
                        )
        return Z

    def backward_pass(self, activation: np.array) -> np.array:
        pass

    def update(self, lr: float) -> None:
        self._W -= lr * self._dW
        self._b -= lr * self._db

    @staticmethod
    def calculate_forward_pass_output_shape(
        activation: np.array,
        W: np.array,
        mode: str
    ) -> Tuple[int, int, int, int]:
        """
        N - number of items
        f - filter count
        fd - filter dimension
        ad - activation dimension
        c - number of channels
        :param activation - activation array with (ad, ad, c, N) shape
        :param W - filter weights array with (f, fd, fd, c) shape
        :param mode - flag describing type of activation padding valid/same
        for example convolution with padding in valid mode, filter shape equal
        to (5, 5, 9), filter count equal to 15 and activation shape equal to
        (11, 11, 9, 64) will produce output with (7, 7, 15, 64)
        """
        f, fd, _, _ = W.shape
        ad, _, _, N = activation.shape
        if mode == 'valid':
            output_dim = ad - fd + 1
            return output_dim, output_dim, f, N
        elif mode == 'same':
            return ad, ad, f, N
        else:
            raise InvalidPaddingModeError(
                f"The padding value can only be equal to valid or same, "
                f"got {mode} instead.")

    @staticmethod
    def pad_activation(
        activation: np.array,
        fd: int,
        mode: str
    ) -> np.array:
        """
        N - number of items
        ad - activation dimension
        c - number of channels
        :param activation - activation array with (ad, ad, c, N) shape
        :param fd - filter dimension
        :param mode - flag describing type of activation padding valid/same
        for example padding in same mode with filter dimension equal to 5 will
        change activation shape from (11, 11, 9, 64) to (15, 15, 9, 64)
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
        activation_slice: np.array,
        filter_W: np.array,
        filter_b: float
    ) -> np.array:
        """
        fd - filter dimension
        c - number of channels
        :param activation_slice - slice of layer activation array with (fd, fd, c) shape
        :param filter_W - single filter weights array with (fd, fd, c) shape
        :param filter_b - single filter bias
        """
        return np.sum(np.multiply(activation_slice, filter_W)) + filter_b
