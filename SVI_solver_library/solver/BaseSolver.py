from abc import ABCMeta, abstractmethod
import torch

class BaseSolver():
    __metaclass__=ABCMeta

    @abstractmethod
    def solve(u: torch.Tensor) -> torch.Tensor:
        """Решить уравнение"""
        pass
