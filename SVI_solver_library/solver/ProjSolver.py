import torch
from SVI_solver_library.solver.BaseSolver import BaseSolver
from SVI_solver_library.transform.BaseTransform import BaseTransform
from SVI_solver_library.integrator.BaseIntegrator import BaseIntegrator

class ProjSolver(BaseSolver):
    def __init__(self, F: BaseTransform, G: BaseTransform, P: BaseTransform,
                eigenvalues: torch.Tensor, integrator: BaseIntegrator):
        self.F = F
        self.G = G
        self.P = P
        self.eigenvalues = eigenvalues
        self.integrator = integrator

    def get_dt(self, u: torch.Tensor):
        f_u = self.F.transform(u)
        g_u = self.G.transform(u)
        return torch.matmul(self.diagonal_matrix, (self.P.transform(g_u-f_u) - g_u))

    def solve(self, u: torch.Tensor) -> torch.Tensor:
        return self.integrator(u, self.get_dt)
