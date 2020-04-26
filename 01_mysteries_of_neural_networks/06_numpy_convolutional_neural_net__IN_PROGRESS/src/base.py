from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

import numpy as np


class Layer(ABC):

    @property
    def weights(self) -> Optional[Tuple[np.array, np.array]]:
        return None

    @property
    def gradients(self) -> Optional[Tuple[np.array, np.array]]:
        return None

    @abstractmethod
    def forward_pass(self, a_prev: np.array) -> np.array:
        pass

    @abstractmethod
    def backward_pass(self, da_curr: np.array) -> np.array:
        pass

    def set_wights(self, w: np.array, b: np.array) -> None:
        pass


class Optimizer(ABC):

    @abstractmethod
    def update(self, layers: List[Layer]) -> None:
        pass

