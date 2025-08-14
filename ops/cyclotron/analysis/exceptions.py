class ConvergenceError(RuntimeError):
    """
    Base class for all convergence failures.
    """
    def __init__(self, *, max_iter: int | None = None,
                 tol: float | None = None):
        super().__init__(f'Routine did not converge after {max_iter} iterations (tolerance = {tol})')
        self.max_iter = max_iter
        self.tol = tol
