import torch
from SVI_solver_library.solver.BaseSolver import AbstractSolver
from SVI_solver_library.transform.BaseTransform import BaseTransform
from SVI_solver_library.integrator.BaseIntegrator import BaseIntegrator

class ProjSolver(BaseSolver):
    def __init__(self, F: BaseTransform, G: BaseTransform, P: BaseTransform,
                eigenvalues: torch.Tensor, integrator: BaseIntegrator):
