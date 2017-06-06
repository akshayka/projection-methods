import numpy as np

from projection_methods.algorithms.optimizer import Optimizer
from projection_methods.algorithms.utils import heavy_ball_update

class AltP(Optimizer):
    def __init__(self,
            max_iters=100, atol=10e-5, initial_iterate=None,
            momentum=None, verbose=False):
        super(AltP, self).__init__(max_iters, atol, initial_iterate, verbose)
        self.momentum = momentum


    def _compute_residual(self, x_k, y_k):
        """Returns dist(x_k, y_k)"""
        return np.linalg.norm(x_k - y_k, 2)

    def _is_optimal(self, r_k):
        return r_k <= self.atol

    def solve(self, problem):
        left_set = problem.sets[0]
        right_set = problem.sets[1]

        iterate = (self._initial_iterate if
            self._initial_iterate is not None else np.ones(problem.dimension))
        iterates = [left_set.project(iterate)]
        residuals = []

        status = Optimizer.Status.INACCURATE
        self.all_iterates = []
        for i in xrange(self.max_iters):
            if self.verbose:
                print 'iteration %d' % i
            x_k = iterates[-1]
            y_k = right_set.project(x_k)
            self.all_iterates.extend([x_k, y_k])

            residuals.append(self._compute_residual(x_k, y_k))
            if self.verbose:
                print '\tresidual: %f' % residuals[-1]
            if self._is_optimal(residuals[-1]):
                status = Optimizer.Status.OPTIMAL
                break
            x_k_plus = left_set.project(y_k)

            if self.momentum is not None:
                iterate = heavy_ball_update(
                    iterates=iterates, velocity=x_k_plus-x_k,
                    alpha=self.momentum['alpha'],
                    beta=self.momentum['beta'])
            iterates.append(x_k_plus)

        return iterates, residuals, status
