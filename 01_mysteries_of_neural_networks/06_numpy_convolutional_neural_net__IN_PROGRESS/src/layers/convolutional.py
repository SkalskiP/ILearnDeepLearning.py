from __future__ import annotations
from src.base import Layer
import numpy as np


class ConvLayer2D(Layer):
    @classmethod
    def initialize(cls) -> ConvLayer2D:
        pass

    def forward_pass(self, activation: np.array) -> np.array:
        pass

    def backward_pass(self, activation: np.array) -> np.array:
        pass

    def update(self, lr: float) -> None:
        pass
