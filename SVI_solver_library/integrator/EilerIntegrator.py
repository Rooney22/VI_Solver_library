from typing import Callable
import torch

class EilerIntegrator():
    def __init__(self, dt: float = 0.01, max_iter: int = 1000):
        self.dt = dt
        self.max_iter = max_iter

    def integrate(self, u: torch.Tensor, dt_getter:
                   Callable[[torch.tensor], torch.tensor]) -> torch.Tensor:
        du_dt = dt_getter(u)
        u_last = u.clone().detach()
        u += self.dt * du_dt
        i = 1
        while i < self.max_iter and torch.sum(abs(u - u_last)) > max_dif:
            du_dt = dt_getter(u)
            u_last = u.clone().detach()
            u += self.dt * du_dt
            i += 1
        return u
