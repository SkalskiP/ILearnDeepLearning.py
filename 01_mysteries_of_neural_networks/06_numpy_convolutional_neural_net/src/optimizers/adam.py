from typing import Tuple, List

import numpy as np

from src.base import Optimizer, Layer


class Adam(Optimizer):
    def __init__(
        self, lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        """
        :param lr - learning rate
        :param beta1 -
        :param beta2 -
        :param eps - small value to avoid zero denominator
        """
        self._cache_v = {}
        self._cache_s = {}
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps

    def update(self, layers: List[Layer]) -> None:
        if len(self._cache_s) == 0 or len(self._cache_v) == 0:
            self._init_cache(layers)

        for idx, layer in enumerate(layers):
            weights, gradients = layer.weights, layer.gradients
            if weights is None or gradients is None:
                continue

            (w, b), (dw, db) = weights, gradients
            dw_key, db_key = Adam._get_cache_keys(idx)

            self._cache_v[dw_key] = self._beta1 * self._cache_v[dw_key] + \
                (1 - self._beta1) * dw
            self._cache_v[db_key] = self._beta1 * self._cache_v[db_key] + \
                (1 - self._beta1) * db

            self._cache_s[dw_key] = self._beta2 * self._cache_s[dw_key] + \
                (1 - self._beta2) * np.square(dw)
            self._cache_s[db_key] = self._beta2 * self._cache_s[db_key] + \
                (1 - self._beta2) * np.square(db)

            dw = self._cache_v[dw_key] / (np.sqrt(self._cache_s[dw_key]) + self._eps)
            db = self._cache_v[db_key] / (np.sqrt(self._cache_s[db_key]) + self._eps)

            layer.set_wights(
                w=w - self._lr * dw,
                b=b - self._lr * db
            )

    def _init_cache(self, layers: List[Layer]) -> None:
        for idx, layer in enumerate(layers):
            gradients = layer.gradients
            if gradients is None:
                continue

            dw, db = gradients
            dw_key, db_key = Adam._get_cache_keys(idx)

            self._cache_v[dw_key] = np.zeros_like(dw)
            self._cache_v[db_key] = np.zeros_like(db)
            self._cache_s[dw_key] = np.zeros_like(dw)
            self._cache_s[db_key] = np.zeros_like(db)

    @staticmethod
    def _get_cache_keys(idx: int) -> Tuple[str, str]:
        """
        :param idx - index of layer
        """
        return f"dw{idx}", f"db{idx}"
