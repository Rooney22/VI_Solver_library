from SVI_solver_library.transform.BaseTransform import BaseTransform
from typing import Callable
import torch

class AdapterTransform(BaseTransform):
    def __init__(self, func: Callable[[torch.Tensor], torch.Tensor]):
        self.func = func
    
    def transform(self, u):
        return self.func(u)
        