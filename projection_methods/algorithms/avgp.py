import numpy as np

from projection_methods.algorithms.optimizer import Optimizer
from projection_methods.algorithms.utils import heavy_ball_update

class AvgP(optimizer.Optimizer):
    def __init__(self,
            max_iters=100, atol=10e-5, initial_iterate=None,
            momentum=None)
        super(AP, self).__init__(max_iters, atol, initial_iterate)
        self.momentum = momentum


    def _compute_residual(self, x_k, y_k, z_k):
        """Returns tuple (dist from left set, dist from right set)"""
        return (np.linalg.norm(x_k - y_k, 2), np.linalg.norm(x_k - z_k, 2))

    def _is_optimal(self, r_k):
        return r_k[0] <= self.atol and r_k[1] <= self.atol

    def solve(self, problem):
        left_set = problem.sets[0]
        right_set = problem.sets[1]

        iterate = (self._initial_iterate if
            self._initial_iterate is not None else np.ones(problem.dimension))
        iterates = [iterate] if self.average
        residuals = []

        status = Optimizer.Status.INACCURATE
        for i in xrange(self.max_iters):
            x_k = self.iterates[-1]
            y_k = left_set.project(x_k)
            z_k = right_set.project(x_k)

            residuals.append(self._compute_residual_avg(x_k, y_k, z_k))
            if self._is_optimal(residuals[-1]):
                status = Optimizer.Status.OPTIMAL
                break
            x_k_plus = 0.5 * (y_k + z_k)

            if self.momentum is not None:
                iterate = heavy_ball_update(
                    iterates=iterates, velocity=x_k_plus-x_k,
                    alpha=self.momentum['alpha'],
                    beta=self.momentum['beta'])
            iterates.append(x_k_plus)

        return iterates, residuals, status
