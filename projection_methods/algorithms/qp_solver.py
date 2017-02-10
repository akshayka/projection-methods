import projection_methods.algorithms.optimizer as optimizer
import projection_methods.algorithms.utils as utils

import numpy as np
import random

class QPSolver(optimizer.Optimizer):
    DISCARD_POLICIES = frozenset(['evict'])


    def __init__(
            self, max_iters=100, eps=10e-5, initial_point=None,
            memory=2, discard_policy='evict', include_probability=1.0,
            momentum=None, search_method=None, average=True):
        """NB: search_method currently unused."""
        optimizer.Optimizer.__init__(self, max_iters, eps, initial_point)
        self.memory = memory
        if include_probability < 0 or include_probability > 1:
            raise ValueError('include_probability must be in [0, 1]')
        self.include_probability = include_probability
        if discard_policy not in QPSolver.DISCARD_POLICIES:
            raise ValueError(
                'Discard policy must be chosen from ' +
                str(QPSolver.DISCARD_POLICIES))
        self.discard_policy = discard_policy
        self.momentum = momentum
        self.search_method = search_method
        self.average = average


    def _containing_halfspace(self, prev_iterate, point, cvx_var):
        normal = prev_iterate - point
        halfspace = (normal.T * (cvx_var - point) <= 0)
        return halfspace


    def solve(self, problem, options={}):
        """problem needs cvx_sets, cvx_vars, var_dim"""
        cvx_sets = problem.cvx_sets
        cvx_var = problem.cvx_var

        # TODO(akshayka): Use row vectors, not column vectors
        iterate = self._initial_point if \
            self._initial_point is not None \
            else np.random.randn(problem.var_dim, 1)
        self.iterates = [iterate]
        self.halfspaces = []
        self.errors = []
        for _ in xrange(self.max_iters):
            prev_iterate = self.iterates[-1]

            # If the target convex sets lived in a 2-dimensional space
            # and were oriented from left-to-right on the plane, we could
            # imagine that one of the sets was to the left of the other;
            # hence, the naming convention below.
            left_point = utils.project(prev_iterate, cvx_sets[0], cvx_var)
            right_point = utils.project(prev_iterate, cvx_sets[1], cvx_var)

            dist_set_one = np.linalg.norm(prev_iterate - left_point)
            dist_set_two = np.linalg.norm(prev_iterate - right_point)
            self.errors.append(dist_set_one + dist_set_two)
            # TODO(akshayka): Termination criterion is incorrect;
            # the "error" calculated here is the distance from the two
            # sets, while it _should_ be the distance from the optimal
            # solution / the intersection of the two sets ...
            if (self.errors[-1] < self.eps):
                break

            if self.include_probability == 1:
                self.halfspaces.append(self._containing_halfspace(
                    prev_iterate=prev_iterate,
                    point=left_point,
                    cvx_var=cvx_var))
                self.halfspaces.append(self._containing_halfspace(
                    prev_iterate=prev_iterate,
                    point=right_point,
                    cvx_var=cvx_var))
            else:
                if random.random() <= self.include_probability:
                    self.halfspaces.append(self._containing_halfspace(
                        prev_iterate=prev_iterate,
                        point=left_point,
                        cvx_var=cvx_var))
                    self.halfspaces.append(self._containing_halfspace(
                        prev_iterate=prev_iterate,
                        point=right_point,
                        cvx_var=cvx_var))

            target_halfspaces = self.halfspaces
            if self.discard_policy == 'evict':
                target_halfspaces = self.halfspaces[-self.memory:]

            # either project onto an outer approximation or take the
            # average of the left and right points
            if len(target_halfspaces) > 0:
                point_to_project = (prev_iterate if not self.average
                    else np.average([left_point, right_point], axis=0))
                iterate = utils.project(point_to_project,
                    target_halfspaces, cvx_var)
            else:
                iterate = np.average([left_point, right_point], axis=0)

            if self.momentum is not None:
                iterate = utils.momentum_update(
                    iterates=self.iterates, velocity=iterate-prev_iterate,
                    alpha=self.momentum['alpha'],
                    beta=self.momentum['beta'])
            self.iterates.append(iterate)
        return self.iterates[-1]
