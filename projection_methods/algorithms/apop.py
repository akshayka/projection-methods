import random

import numpy as np

from projection_methods.algorithms.optimizer import Optimizer
from projection_methods.algorithms.utils import (heavy_ball_update,
                                                relax, rec_sum)
from projection_methods.oracles.affine_set import AffineSet
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
            max_iters=100, atol=10e-8, do_all_iters=False,
            initial_iterate=None,
            outer_policy=PolyOuter.EXACT,
            max_hyperplanes=None, max_halfspaces=None,
            data_hyperplanes=0, affine_policy='random',
            info=[],
            momentum=None, average=True, theta=1.0,
            verbose=False):
        super(APOP, self).__init__(max_iters, atol, do_all_iters,
            initial_iterate, verbose)
        if outer_policy not in PolyOuter.POLICIES:
            raise ValueError(
                'Policy must be chosen from ' +
                str(PolyOuter.DISCARD_POLICIES))
        self.outer_policy = outer_policy
        self.max_hyperplanes = (max_hyperplanes if max_hyperplanes is not None
            else float('inf'))
        self.max_halfspaces = (max_halfspaces if max_halfspaces is not None
            else float('inf'))
        self.data_hyperplanes = data_hyperplanes
        self.affine_policy = affine_policy
        self.info = info
        self.momentum = momentum
        if theta <= 0 or theta >= 2:
            raise ValueError('relaxation parameter must be in (0, 2); '
                'received %f' % theta)
        self.theta = theta
        self.average = average


    def _verbose_residual(self, x_k, r, fejer_r, left_set, right_set):
        """Print verbose info about the residual r if verbosity is set"""
        if self.verbose:
            print '\tproblem residual: %s (sum %e)' % (str(r), sum(e))
            print '\tresidual w.r.t. opt: %e' % fejer_r
            print '\t\tleft (%s): %e' % (type(left_set).__name__, r[0])
            left_res = '\n'.join([
                '\t\t\t' + l for l in
                left_set.residual_str(x_k).split('\n')])
            print '\t\tresidual breakdown\n%s' % left_res
            print '\t\t\tsum: %e' % rec_sum(left_set.residual(x_k))
            print '\t\tright (%s): %e' % (type(right_set).__name__, r[1])
            right_res = '\n'.join([
                '\t\t\t' + l for l in
                right_set.residual_str(x_k).split('\n')])
            print '\t\tresidual breakdown\n%s' % right_res
            print '\t\t\tsum: %e' % rec_sum(right_set.residual(x_k))


    def _generate_information(self, x_k, left_set, right_set):
        """Carry out intermediate per-iteration computations

        Generate new information to add to our outer approximation,
        and consequentially produce an intermediate iterate x_k_prime, the
        point that will be projected upon the outer approximation to obtain the
        subsequent bonafide iterate, x_k_plus. Produce, too, points y_k =
        left_set.project(x_k), z_k = right_set.project(x_k) for residual
        computation.

        Args:
            x_k (numpy.ndarray): the current iterate
            left_set (Oracle): first of the two oracles to query
            right_set (Oracle): second of the two oracles to query
        Returns:
           numpy.ndarray: the intermediate iterate x_k_prime
           list of Halfspace/Hyperplane: the new information generated
        """
        if self.average:
            if self.verbose:
                print 'performing _averaged_ round'
                print '\tprojecting onto right set ...'
            y_k, y_h_k = self._right_set_query(x_k)
            if self.verbose:
                print '\tprojecting onto left set ...'
            z_k, z_h_k = self._left_set_query(x_k)
            # TODO(akshayka): consider taking x_k_prime to simply be x_k
            x_k_prime = 0.5 * (y_k + z_k)
        else:
            if self.verbose:
                print 'performing _alternating_ round'
                print '\tprojecting onto right set ...'
            y_k, y_h_k = self._right_set_query(x_k)
            if self.verbose:
                print '\tprojecting onto left set ...'
            z_k, z_h_k = self._left_set_query(y_k)
            x_k_prime = z_k
        return x_k_prime, y_h_k + z_h_k


    def _query_func(self, oracle):
        if self.data_hyperplanes > 0 and isinstance(oracle, AffineSet):
            return lambda x: oracle.query(x,
                data_hyperplanes=self.data_hyperplanes,
                policy=self.affine_policy)
        else:
            return lambda x: oracle.query(x)


    def _push_residuals(self, problem, x_k_prime, residuals, fejer_residuals,
            left_set, right_set):
        r = problem.residual(x_k_prime)
        residuals.append(r)
        fejer_r = np.linalg.norm(x_k_prime - problem.x_opt, 2)
        fejer_residuals.append(fejer_r)
        self._verbose_residual(x_k_prime, r, fejer_r, left_set, right_set)


    def solve(self, problem):
        # the nomenclature `left` and `right` is by my convention;
        # the picture to have in mind is two sets in R^2, one to the left
        # of the other.
        left_set = problem.sets[0]
        self._left_set_query = self._query_func(left_set)
        right_set = problem.sets[1]
        self._right_set_query = self._query_func(right_set)

        outer = left_set.outer(kind=ConvexOuter.EMPTY)
        assert len(outer.hyperplanes()) == 0
        assert len(outer.halfspaces()) == 0
        self.outer_manager = DynamicPolyhedron(polyhedron=outer, 
            max_hyperplanes=self.max_hyperplanes,
            max_halfspaces=self.max_halfspaces, policy=self.outer_policy)
        self.outer_manager.add(self.info)
        
        iterate = (self._initial_iterate if
            self._initial_iterate is not None else np.ones(problem.dimension))
        iterates = [iterate]
        residuals = []
        fejer_residuals = []
        self._push_residuals(problem, iterate, residuals, fejer_residuals,
            left_set, right_set)

        status = Optimizer.Status.INACCURATE
        for i in xrange(self.max_iters):
            if self.verbose:
                print 'iteration %d' % i
            # Execute the intermediate step.
            x_k_prime, info = self._generate_information(iterates[-1],
                left_set, right_set)

            # Compute residuals for x_k_prime
            self._push_residuals(problem, x_k_prime, residuals,
                fejer_residuals, left_set, right_set)
            if self._is_optimal(residuals[-1]):
                status = Optimizer.Status.OPTIMAL
                if not self.do_all_iters:
                    break

            self.outer_manager.add(info)
            outer = self.outer_manager.outer()
            if self.verbose:
                print '\tobtained %d pieces of information' % len(info)
                print '\tprojecting onto outer approximation with ...'
                print '\t\t%d hyperplanes' % len(outer.hyperplanes())
                print '\t\t%d halfspaces' % len(outer.halfspaces())
            x_k_plus = outer.project(x_k_prime)

            # Debugging: Check whether the sequence produced by APOP violates
            # fejer monotonicity (w.r.t. a single optimal point problem.x_opt)
            fejer_r = fejer_residuals[-1]
            next_fejer_r = np.linalg.norm(x_k_plus - problem.x_opt, 2)
            if next_fejer_r > fejer_r:
                raise RuntimeError('Localization step is not Fejer monotonic;'
                    'residual increased from %e to %e' % (
                    fejer_r, next_fejer_r))
            else:
                print '\fejer residual delta %e' % (next_fejer_r -
                    fejer_r)

            # Relax the new iterate and/or take into account previous iterates'
            # momentum.
            if self.theta != 1.0:
                x_k_plus = relax(x_k_prime, x_k_plus, self.theta)
            if self.momentum is not None:
                x_k_plus = heavy_ball_update(
                    iterates=iterates, velocity=x_k_plus-x_k,
                    alpha=self.momentum[0],
                    beta=self.momentum[1])
            iterates.append(x_k_plus)
        return iterates, residuals, status
