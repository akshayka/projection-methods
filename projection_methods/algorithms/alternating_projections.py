from projection_methods.algorithms.optimizer import Optimizer
from projection_methods.algorithms.utils import project

import numpy as np

class AlternatingProjections(Optimizer):

    def solve(self, problem, options={}):
        """problem needs cvx_sets, cvx_vars, var_dim"""
        cvx_sets = problem.cvx_sets
        cvx_var = problem.cvx_var

        # TODO(akshayka): Smarter selection of initial iterate
        iterate = options['initial_point'] if \
            options.get('initial_point') is not None \
            else np.random.randn(problem.var_dim, 1) 
        iterates = [iterate]
        self.errors = []
        for i in xrange(self.max_iters):
            prev_iterate = iterates[-1]
            iterate = project(prev_iterate, [cvx_sets[i % 2]], cvx_var)
            dist = np.linalg.norm(prev_iterate - iterate)
            self.errors.append(dist)
            if (dist < self.eps):
                break
            else:
                iterates.append(iterate)
        return iterates
