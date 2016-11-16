import projection_methods.algorithms.optimizer as optimizer
import projection_methods.algorithms.utils as utils

import numpy as np

class Dykstra(optimizer.Optimizer):
    def __init__(self, max_iters=100, eps=10e-5, initial_point=None):
        optimizer.Optimizer.__init__(self, max_iters, eps, initial_point)


    def solve(self, problem, options={}):
        """problem needs cvx_sets, cvx_vars, var_dim"""
        cvx_sets = problem.cvx_sets
        cvx_var = problem.cvx_var

        if self._initial_point is None:
            self._initial_point = np.random.randn(problem.var_dim, 1) 

        # (p_n), (q_n) are the auxiliary sequences, (a_n), (b_n) the main
        # sequences, defined in Bauschke's 98 paper
        # (Dykstra's Alternating Projection Algorithm for Two Sets).
        self.p = [None] * (self.max_iters + 1)
        self.q = [None] * (self.max_iters + 1)
        self.a = [None] * (self.max_iters + 1)
        self.b = [None] * (self.max_iters + 1)
        zero_vector = np.zeros(shape=(problem.var_dim, 1))
        self.p[0] = self.q[0] = zero_vector
        self.b[0] = self.a[0] = self._initial_point
        for n in xrange(1, self.max_iters + 1):
            # TODO(akshayka): Termination criterion
            self.a[n] = utils.project(
                self.b[n-1] + self.p[n-1], cvx_sets[0], cvx_var)
            self.b[n] = utils.project(
                self.a[n] + self.q[n-1], cvx_sets[1], cvx_var)
            self.p[n] = self.b[n-1] + self.p[n-1] - self.a[n]
            self.q[n] = self.a[n] + self.q[n-1] - self.b[n]
        self.iterates = self.a
        return self.a[-1]
