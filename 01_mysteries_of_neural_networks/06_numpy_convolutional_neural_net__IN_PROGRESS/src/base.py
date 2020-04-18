from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    @abstractmethod
    def forward_pass(self, activation: np.array) -> np.array:
        pass

    @abstractmethod
    def backward_pass(self, activation: np.array) -> np.array:
        pass

    @abstractmethod
    def update(self, lr: float) -> None:
        pass
