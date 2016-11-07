from projection_methods.algorithms.optimizer import Optimizer
from projection_methods.algorithms.utils import project

import numpy as np

class AlternatingProjections(Optimizer):
    MAX_ITERS = int(1e2)

    def solve(self, problem, options={}):
        """problem needs cvx_sets, cvx_vars, var_dim"""
        cvx_sets = problem.cvx_sets
        cvx_var = problem.cvx_var

        # TODO(akshayka): Smarter selection of initial iterate
        iterate = options['initial_point'] if \
            options.get('initial_point') is not None \
            else np.random.randn(problem.var_dim, 1) 
        max_iters = options['max_iters'] if \
            options.get('max_iters') is not None \
            else AlternatingProjections.MAX_ITERS
        iterates = [iterate]
        # TODO(akshayka): Termination criteria.
        for _ in xrange(max_iters):
            prev_iterate = iterates[-1]
            first_iterate = project(prev_iterate, [cvx_sets[0]], cvx_var)
            if (np.allclose(prev_iterate, first_iterate)):
                break
            iterates.append(first_iterate)
            second_iterate = project(first_iterate, [cvx_sets[1]], cvx_var)

            if (np.allclose(first_iterate, second_iterate)):
                break
            iterates.append(second_iterate)
        return iterates
