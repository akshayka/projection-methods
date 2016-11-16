import projection_methods.algorithms.optimizer as optimizer
import projection_methods.algorithms.utils as utils

import numpy as np

class AlternatingProjections(optimizer.Optimizer):
    def __init__(
            self, max_iters=100, eps=10e-5, initial_point=None,
            momentum=None):
        optimizer.Optimizer.__init__(self, max_iters, eps, initial_point)
        self.momentum = momentum


    def solve(self, problem, options={}):
        """problem needs cvx_sets, cvx_vars, var_dim"""
        cvx_sets = problem.cvx_sets
        cvx_var = problem.cvx_var

        # TODO(akshayka): Smarter selection of initial iterate
        iterate = self._initial_point if \
            self._initial_point is not None \
            else np.random.randn(problem.var_dim, 1) 
        self.iterates = [iterate]
        self.errors = []
        for i in xrange(self.max_iters):
            prev_iterate = self.iterates[-1]
            iterate = utils.project(prev_iterate, cvx_sets[i % 2], cvx_var)
            if self.momentum is not None:
                iterate = utils.momentum_update(
                    iterates=self.iterates, velocity=iterate-prev_iterate,
                    alpha=self.momentum['alpha'],
                    beta=self.momentum['beta'])
            dist = np.linalg.norm(prev_iterate - iterate)
            self.errors.append(dist)
            if (dist < self.eps):
                break
            else:
                self.iterates.append(iterate)
        return self.iterates[-1]
