from __future__ import annotations

from typing import Tuple

import numpy as np

from src.base import Layer


class MaxPoolLayer(Layer):
    def __init__(self, pool_size: Tuple[int, int], strides: int = 2):
        self._pool_size = pool_size
        self._strides = strides
        self._A = None

    def forward_pass(self, activation: np.array) -> np.array:
        """
        N - number of items
        H, W - activation dimensions
        C - number of channels
        :param activation - activation array with (H, W, C, N) shape
        """
        self._A = np.array(activation, copy=True)
        H, W, C, N = activation.shape
        HH, WW = self._pool_size

        assert (H - HH) % self._strides == 0
        assert (W - WW) % self._strides == 0

        H_prime = 1 + (H - HH) // self._strides
        W_prime = 1 + (W - WW) // self._strides

        output = np.zeros((H_prime, W_prime, C, N))

        for n in range(N):
            for j in range(H_prime):
                for i in range(W_prime):
                    for c in range(C):
                        h_start = j * self._strides
                        h_end = h_start + HH
                        w_start = i * self._strides
                        w_end = w_start + WW
                        output[j, i, c, n] = np.max(activation[h_start:h_end, w_start:w_end, c, n])
        return output

    def backward_pass(self, activation: np.array) -> np.array:
        H, W, C, N = self._A.shape
        HH, WW = self._pool_size

        H_prime = 1 + (H - HH) // self._strides
        W_prime = 1 + (W - WW) // self._strides

        output = np.zeros_like(self._A)

        for n in range(N):
            for j in range(H_prime):
                for i in range(W_prime):
                    for c in range(C):
                        h_start = j * self._strides
                        h_end = h_start + HH
                        w_start = i * self._strides
                        w_end = w_start + WW
                        idx = np.argmax(self._A[h_start:h_end, w_start:w_end, c, n])
                        idx_h, idx_w = np.unravel_index(idx, (HH, WW))
                        output[h_start:h_end, w_start:w_end, c, n][idx_h, idx_w] = activation[j, i, c, n]
        return output

