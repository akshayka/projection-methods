import numpy as np

from projection_methods.algorithms.optimizer import Optimizer

class Dykstra(Optimizer):
    def __init__(self,
            max_iters=100, atol=10e-5, initial_iterate=None, verbose=False):
        super(Dykstra, self).__init__(max_iters, atol, initial_iterate, verbose)


    def _compute_residual(self, x_k, left, right):
        """Returns tuple (dist from left set, dist from right set)"""
        y_k = left.project(x_k)
        z_k = right.project(x_k)
        return (np.linalg.norm(x_k - y_k, 2), np.linalg.norm(x_k - z_k, 2))

    def _is_optimal(self, r_k):
        return r_k[0] <= self.atol and r_k[1] <= self.atol

    def solve(self, problem):
        left_set = problem.sets[0]
        right_set = problem.sets[1]

        iterate = (self._initial_iterate if
            self._initial_iterate is not None else np.ones(problem.dimension))

        # (p_n), (q_n) are the auxiliary sequences, (a_n), (b_n) the main
        # sequences, defined in Bauschke's 98 paper
        # (Dykstra's Alternating Projection Algorithm for Two Sets).
        self.p = [None] * (self.max_iters + 1)
        self.q = [None] * (self.max_iters + 1)
        self.a = [None] * (self.max_iters + 1)
        self.b = [None] * (self.max_iters + 1)
        zero_vector = np.zeros(problem.dimension)
        self.p[0] = self.q[0] = zero_vector
        self.b[0] = self.a[0] = iterate
        residuals = []

        status = Optimizer.Status.INACCURATE
        for n in xrange(1, self.max_iters + 1):
            if self.verbose:
                print 'iteration %d' % n
            # TODO(akshayka): Robust stopping criterion
            residuals.append(self._compute_residual(
                self.b[n-1], left_set, right_set))
            if self._is_optimal(residuals[-1]):
                status = Optimizer.Status.OPTIMAL
                break

            self.a[n] = left_set.project(self.b[n-1] + self.p[n-1])
            self.b[n] = right_set.project(self.a[n] + self.q[n-1])
            self.p[n] = self.b[n-1] + self.p[n-1] - self.a[n]
            self.q[n] = self.a[n] + self.q[n-1] - self.b[n]

        return self.b, residuals, status
