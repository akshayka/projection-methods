import projection_methods.algorithms.optimizer as optimizer
import projection_methods.algorithms.utils as utils

import numpy as np

class AlternatingProjections(optimizer.Optimizer):
    def __init__(
            self, max_iters=100, eps=10e-5, initial_point=None,
            momentum=None, plane_search=[1,1]):
        """
        plane_search[i] == number of iterates of the i-th set to include
            in the plane search. 1 ==> no plane search.
        """
        optimizer.Optimizer.__init__(self, max_iters, eps, initial_point)
        self.momentum = momentum
        self.plane_search = plane_search


    def _iterates_for_set(self, set_idx):
        # recall that the initial point is not in either set
        return self.iterates[1+set_idx::2]


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
            # the index of the convex set to project on
            target_set_idx = i % 2
            # the index of the convex set projected on in the previous iteration
            other_set_idx = (i - 1) % 2

            # Instead of projecting the previous iterate onto the convex
            # set, perform a plane search of previous iterates on the other
            # set. 
            plane_search_param = self.plane_search[target_set_idx]
            iterates_for_set = self._iterates_for_set(other_set_idx)
            if plane_search_param > 1 and len(iterates_for_set) > 1:
                # TODO(akshayka): other set idx?
                iterate, _ = utils.plane_search(
                    iterates_for_set, plane_search_param,
                    cvx_sets[target_set_idx], cvx_var)
            else:
                iterate = utils.project(
                    prev_iterate, cvx_sets[target_set_idx], cvx_var)
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
