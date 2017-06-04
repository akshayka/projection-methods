import random

import numpy as np

from projection_methods.algorithms.optimizer import Optimizer
from projection_methods.algorithms.utils import heavy_ball_update
from projection_methods.oracles.convex_set import ConvexOuter
from projection_methods.oracles.dynamic_polyhedron import DynamicPolyhedron
from projection_methods.oracles.dynamic_polyhedron import PolyOuter

class APOP(Optimizer):
    """Alternating Projections (accelerated by) Outer Approximations

    TODO(akshayka):
        line/plane search
        more fine-grained residuals (primal/dual)
        over/under-projection (two or three state variables instead of just
            one per iteration, a la ADMM)
    """
    def __init__(self,
            max_iters=100, atol=10e-5, initial_iterate=None,
            outer_policy=PolyOuter.EXACT,
            max_hyperplanes=float("inf"), max_halfspaces=float("inf"),
            momentum=None, average=True):
        super(APOP, self).__init__(max_iters, atol, initial_iterate)
        if outer_policy not in PolyOuter.POLICIES:
            raise ValueError(
                'Policy must be chosen from ' +
                str(PolyOuter.DISCARD_POLICIES))
        self.outer_policy = outer_policy
        self.max_hyperplanes = max_hyperplanes
        self.max_halfspaces = max_halfspaces
        self.momentum = momentum
        self.average = average
        self.iterates = []
        self.residuals = []


    def _compute_residual(self, x_k, y_k, z_k):
        """Returns tuple (dist from left set, dist from right set)"""
        return (np.linalg.norm(x_k - y_k, 2), np.linalg.norm(x_k - z_k, 2))

    def _is_optimal(self, r_k):
        return r_k[0] <= self.atol and r_k[1] <= self.atol

    def solve(self, problem):
        # the nomenclature `left` and `right` is by my convention;
        # the picture to have in mind is two sets in R^2, one to the left
        # of the other.
        left_set = problem.sets[0]
        right_set = problem.sets[1]

        outer = left_set.outer(kind=ConvexOuter.POLYHEDRAL)
        assert len(outer.hyperplanes()) == 0
        assert len(outer.halfspaces()) == 0
        self.outer_manager = DynamicPolyhedron(polyhedron=outer, 
            max_hyperplanes=self.max_hyperplanes,
            max_halfspaces=self.max_halfspaces, policy=self.outer_policy)
        
        # TODO(akshayka): more intelligent selection of the initial iterate;
        # in particular, if solving a self-dual homogeneous embedding, we
        # must avoid convergence to zero.
        iterate = (self._initial_iterate if
            self._initial_iterate is not None else np.ones(problem.dimension))
        self.iterates = [iterate]

        status = Optimizer.Status.INACCURATE
        for _ in xrange(self.max_iters):
            print "iteration %d" % _
            x_k = self.iterates[-1]
            y_k, y_h_k = left_set.query(x_k)
            z_k, z_h_k = right_set.query(x_k)
            self.residuals.append(self._compute_residual(x_k, y_k, z_k))
            if self._is_optimal(self.residuals[-1]):
                status = Optimizer.Status.OPTIMAL
                break

            self.outer_manager.add([y_h_k, z_h_k])
            if self.average:
                x_k_plus = self.outer_manager.outer().project((y_k + z_k) / 2.0)
            else:
                x_k_plus = self.outer_manager.outer().project(x_k)

            if self.momentum is not None:
                iterate = momentum_update(
                    iterates=self.iterates, velocity=x_k_plus-x_k,
                    alpha=self.momentum['alpha'],
                    beta=self.momentum['beta'])
            self.iterates.append(x_k_plus)

        return self.iterates, self.residuals, status
