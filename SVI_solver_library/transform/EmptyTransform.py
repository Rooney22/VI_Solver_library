from SVI_solver_library.transform.BaseTransform import BaseTransform

class EmptyTransform(BaseTransform):
    def transform(self, u):
        return u
        