import numpy as np

from projection_methods.algorithms.optimizer import Optimizer
from projection_methods.algorithms.utils import heavy_ball_update

class AvgP(Optimizer):
    def __init__(self,
            max_iters=100, atol=10e-5, do_all_iters=False,
            initial_iterate=None, momentum=None, verbose=False):
        super(AvgP, self).__init__(max_iters, atol, do_all_iters,
            initial_iterate, verbose)
        self.momentum = momentum


    def solve(self, problem):
        left_set = problem.sets[0]
        right_set = problem.sets[1]

        iterate = (self._initial_iterate if
            self._initial_iterate is not None else np.ones(problem.dimension))
        iterates = [iterate]
        residuals = []

        status = Optimizer.Status.INACCURATE
        for i in xrange(self.max_iters):
            if self.verbose:
                print 'iteration %d' % i
            x_k = iterates[-1]
            y_k = left_set.project(x_k)
            z_k = right_set.project(x_k)

            residuals.append(self._compute_residual(x_k, y_k, z_k))
            if self.verbose:
                print '\tresidual: %e' % sum(residuals[-1])
            if self._is_optimal(residuals[-1]):
                status = Optimizer.Status.OPTIMAL
                if not self.do_all_iters:
                    break
            x_k_plus = 0.5 * (y_k + z_k)

            if self.momentum is not None:
                iterate = heavy_ball_update(
                    iterates=iterates, velocity=x_k_plus-x_k,
                    alpha=self.momentum['alpha'],
                    beta=self.momentum['beta'])
            iterates.append(x_k_plus)

        return iterates, residuals, status
