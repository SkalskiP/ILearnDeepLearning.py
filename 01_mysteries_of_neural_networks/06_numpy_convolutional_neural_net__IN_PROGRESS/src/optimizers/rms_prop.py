from typing import List, Tuple

import numpy as np

from src.base import Optimizer, Layer


class RMSProp(Optimizer):
    def __init__(self, lr: float, beta: float = 0.9, eps: float = 1e-8):
        """
        :param lr - learning rate
        :param beta - discounting factor for the history/coming gradient
        :param eps - small value to avoid zero denominator
        """
        self._cache = {}
        self._lr = lr
        self._beta = beta
        self._eps = eps

    def update(self, layers: List[Layer]) -> None:
        if len(self._cache) == 0:
            self._init_cache(layers)

        for idx, layer in enumerate(layers):
            weights, gradients = layer.weights, layer.gradients
            if weights is None or gradients is None:
                continue

            (w, b), (dw, db) = weights, gradients
            dw_key, db_key = RMSProp._get_cache_keys(idx)

            self._cache[dw_key] = self._beta * self._cache[dw_key] + \
                (1 - self._beta) * np.square(dw)
            self._cache[db_key] = self._beta * self._cache[db_key] + \
                (1 - self._beta) * np.square(db)

            dw = dw / (np.sqrt(self._cache[dw_key]) + self._eps)
            db = db / (np.sqrt(self._cache[db_key]) + self._eps)

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
            dw_key, db_key = RMSProp._get_cache_keys(idx)

            self._cache[dw_key] = np.zeros_like(dw)
            self._cache[db_key] = np.zeros_like(db)

    @staticmethod
    def _get_cache_keys(idx: int) -> Tuple[str, str]:
        """
        :param idx - index of layer
        """
        return f"dw{idx}", f"db{idx}"

