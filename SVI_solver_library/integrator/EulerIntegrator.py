from typing import Callable
import torch

class EulerIntegrator():
    def __init__(self, dt: float = 0.01, max_iter: int = 1000, max_dif: float = 1e-4):
        self.dt = dt
        self.max_iter = max_iter
        self.max_dif = max_dif

    def integrate(self, u: torch.Tensor, du_dt_getter:
                   Callable[[torch.tensor], torch.tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        du_dt = du_dt_getter(u)
        u_last = u.clone().detach()
        u += self.dt * du_dt
        verbose_data = u.clone().detach().unsqueeze(0)
        i = 1
        while i < self.max_iter and torch.sum(abs(u - u_last)) > self.max_dif:
            verbose_data = torch.cat((verbose_data, u.clone().detach().unsqueeze(0)), dim=0)
            du_dt = du_dt_getter(u)
            u_last = u.clone().detach()
            u += self.dt * du_dt
            i += 1
        return (u, verbose_data)
