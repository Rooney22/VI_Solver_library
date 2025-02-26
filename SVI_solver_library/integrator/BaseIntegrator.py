from abc import ABCMeta, abstractmethod
from typing import Callable
import torch

class BaseIntegrator():
    __metaclass__=ABCMeta

    @abstractmethod
    def integrate(u: torch.Tensor, dt_getter: Callable[[torch.tensor], torch.tensor]) -> torch.Tensor:
        """Интегрирует"""
        pass
