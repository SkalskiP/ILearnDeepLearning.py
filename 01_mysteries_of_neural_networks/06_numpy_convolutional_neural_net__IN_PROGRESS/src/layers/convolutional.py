from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

from src.base import Layer
from src.errors import InvalidPaddingModeError


class ConvLayer2D(Layer):
    def __init__(
        self, w: np.array,
        b: np.array,
        padding: str = 'valid',
        stride: int = 2
    ):
        """
        :param w -  4D tensor with shape (h_f, w_f, c_f, n_f)
        :param b - 1D tensor with shape (n_f, )
        :param padding - flag describing type of activation padding valid/same
        :param stride - stride along width and height of input volume used to
        ------------------------------------------------------------------------
        h_f - height of filter volume
        w_f - width of filter volume
        c_f - number of channels of filter volume
        n_f - number of filters in filter volume
        """
        self._w, self._b = w, b
        self._padding = padding
        self._stride = stride
        self._dw, self._db = None, None
        self._A = None

    @classmethod
    def initialize(
        cls, filters: int,
        kernel_shape: Tuple[int, int, int],
        padding: str = 'valid',
        stride: int = 2
    ) -> ConvLayer2D:
        w = np.random.randn(*kernel_shape, filters) * 0.1
        b = np.random.randn(filters) * 0.1
        return cls(w=w, b=b, padding=padding, stride=stride)

    @property
    def weights(self) -> Optional[Tuple[np.array, np.array]]:
        return self._w, self._b

    @property
    def gradients(self) -> Optional[Tuple[np.array, np.array]]:
        if self._dw is None or self._db is None:
            return None
        return self._dw, self._db

    def forward_pass(self, a_prev: np.array) -> np.array:
        """
        :param a_prev - 4D tensor with shape(n, h, w, c)
        :output 4D tensor with shape(n, h, w, n_f)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w - width of input volume
        h - width of input volume
        c - number of channels of the input volume
        n_f - number of filters in filter volume
        """
        pass

    def backward_pass(self, da_curr: np.array) -> np.array:
        """
        :param da_curr - 4D tensor with shape(n, h, w, n_f)
        :output 4D tensor with shape(n, h, w, c)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w - width of input volume
        h - width of input volume
        c - number of channels of the input volume
        n_f - number of filters in filter volume
        """
        pass

    def set_wights(self, w: np.array, b: np.array) -> None:
        """
        :param w -  4D tensor with shape (h_f, w_f, c_f, n_f)
        :param b - 1D tensor with shape (n_f, )
        ------------------------------------------------------------------------
        h_f - height of filter volume
        w_f - width of filter volume
        c_f - number of channels of filter volume
        n_f - number of filters in filter volume
        """
        self._w = w
        self._b = b

    def get_pad_width(self) -> int:
        if self._padding == 'same':
            return int((self._w.shape[0] - 1) / 2)
        elif self._padding == 'valid':
            return 0
        else:
            raise InvalidPaddingModeError(
                f"Unsupported padding value: {self._padding}"
            )

    @staticmethod
    def pad(array: np.array, pad: int) -> np.array:
        return np.pad(
            array=array,
            pad_width=((0, 0), (pad, pad), (pad, pad), (0, 0)),
            mode='constant'
        )
