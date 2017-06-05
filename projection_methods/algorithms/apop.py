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
            max_hyperplanes=None, max_halfspaces=None,
            momentum=None, average=True, verbose=False):
        super(APOP, self).__init__(max_iters, atol, initial_iterate, verbose)
        if outer_policy not in PolyOuter.POLICIES:
            raise ValueError(
                'Policy must be chosen from ' +
                str(PolyOuter.DISCARD_POLICIES))
        self.outer_policy = outer_policy
        self.max_hyperplanes = (max_hyperplanes if max_hyperplanes is not None
            else float('inf'))
        self.max_halfspaces = (max_halfspaces if max_halfspaces is not None
            else float('inf'))
        self.momentum = momentum
        self.average = average


    def _compute_residual_aux(self, x_k, y_k, z_k):
        """Returns tuple (dist from left set, dist from right set)"""
        return (np.linalg.norm(x_k - y_k, 2), np.linalg.norm(x_k - z_k, 2))

    def _compute_residual(self, x_k, left_set, right_set):
        """Returns tuple (dist from left set, dist from right set)"""
        y_k, _ = left_set.query(x_k)
        z_k, _ = right_set.query(x_k)
        return self._compute_residual_aux(x_k, y_k, z_k)

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
        iterates = [iterate]
        residuals = []

        status = Optimizer.Status.INACCURATE
        for i in xrange(self.max_iters):
            if self.verbose:
                print 'iteration %d' % i
            x_k = iterates[-1]
            y_k, y_h_k = left_set.query(x_k)
            z_k, z_h_k = right_set.query(x_k)
            residuals.append(self._compute_residual_aux(x_k, y_k, z_k))
            if self._is_optimal(residuals[-1]):
                status = Optimizer.Status.OPTIMAL
                break

            self.outer_manager.add(y_h_k + z_h_k)
            if self.average:
                x_k_plus = self.outer_manager.outer().project(0.5 * (y_k + z_k))
            else:
                x_k_plus = self.outer_manager.outer().project(x_k)

            if self.momentum is not None:
                iterate = heavy_ball_update(
                    iterates=iterates, velocity=x_k_plus-x_k,
                    alpha=self.momentum['alpha'],
                    beta=self.momentum['beta'])
            iterates.append(x_k_plus)

        return iterates, residuals, status
