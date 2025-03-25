import torch
from SVI_solver_library.solver.BaseSolver import BaseSolver
from SVI_solver_library.transform.BaseTransform import BaseTransform
from SVI_solver_library.integrator.BaseIntegrator import BaseIntegrator
from SVI_solver_library.integrator.EulerIntegrator import EulerIntegrator
from SVI_solver_library.transform.EmptyTransform import EmptyTransform

class ProjSolver(BaseSolver):
    def __init__(self, F: BaseTransform, eigenvalues: torch.Tensor = None, P: BaseTransform = EmptyTransform(),
                  G: BaseTransform = EmptyTransform(), integrator: BaseIntegrator = EulerIntegrator(),
                  verbose: bool = True):
        self.F = F
        self.G = G
        self.P = P
        if eigenvalues is None:
            self.diagonal_matrix = None
        else:
            self.diagonal_matrix = torch.diag(eigenvalues)
        self.integrator = integrator

    def get_du_dt(self, u: torch.Tensor):
        if self.diagonal_matrix == None:
            self.diagonal_matrix = torch.diag(torch.full((len(u),), 4, dtype=torch.double))
        f_u = self.F.transform(u)
        g_u = self.G.transform(u)
        return torch.matmul(self.diagonal_matrix, (self.P.transform(g_u-f_u) - g_u))

    def solve(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.integrator.integrate(u, self.get_du_dt)
