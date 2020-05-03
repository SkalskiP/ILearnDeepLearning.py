from __future__ import annotations

from typing import Tuple

import numpy as np

from src.base import Layer


class MaxPoolLayer(Layer):

    def __init__(self, pool_size: Tuple[int, int], stride: int = 2):
        """
        :param pool_size - tuple holding shape of 2D pooling window
        :param stride - stride along width and height of input volume used to
        apply pooling operation
        """
        self._pool_size = pool_size
        self._stride = stride
        self._a = None
        self._cache = {}

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        """
        :param a_prev - 4D tensor with shape(n, h_in, w_in, c)
        :output 4D tensor with shape(n, h_out, w_out, c)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        c - number of channels of the input/output volume
        w_out - width of output volume
        h_out - width of output volume
        """
        self._a = np.array(a_prev, copy=True)
        n, h_in, w_in, c = a_prev.shape
        h_pool, w_pool = self._pool_size
        h_out = 1 + (h_in - h_pool) // self._stride
        w_out = 1 + (w_in - w_pool) // self._stride
        output = np.zeros((n, h_out, w_out, c))

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_pool
                w_start = j * self._stride
                w_end = w_start + w_pool
                a_prev_slice = a_prev[:, h_start:h_end, w_start:w_end, :]
                self._save_mask(x=a_prev_slice, cords=(i, j))
                output[:, i, j, :] = np.max(a_prev_slice, axis=(1, 2))
        return output

    def backward_pass(self, da_curr: np.array) -> np.array:
        """
        :param da_curr - 4D tensor with shape(n, h_out, w_out, c)
        :output 4D tensor with shape(n, h_in, w_in, c)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        c - number of channels of the input/output volume
        w_out - width of output volume
        h_out - width of output volume
        """
        output = np.zeros_like(self._a)
        _, h_out, w_out, _ = da_curr.shape
        h_pool, w_pool = self._pool_size

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_pool
                w_start = j * self._stride
                w_end = w_start + w_pool
                output[:, h_start:h_end, w_start:w_end, :] += \
                    da_curr[:, i:i + 1, j:j + 1, :] * self._cache[(i, j)]
        return output

    def _save_mask(self, x: np.array, cords: Tuple[int, int]) -> None:
        mask = np.zeros_like(x)
        n, h, w, c = x.shape
        x = x.reshape(n, h * w, c)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((n, c))
        mask.reshape(n, h * w, c)[n_idx, idx, c_idx] = 1
        self._cache[cords] = mask


