from abc import ABCMeta, abstractmethod
import torch

class BaseTransform():
    __metaclass__=ABCMeta

    @abstractmethod
    def transform(u: torch.tensor) -> torch.tensor:
        """Трансформировать"""
        pass
