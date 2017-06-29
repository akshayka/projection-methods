import numpy as np

from projection_methods.algorithms.optimizer import Optimizer
from projection_methods.problems.problems import SCSProblem

class SCSADMM(Optimizer):
    def __init__(self,
            max_iters=100, atol=10e-5, do_all_iters=False, verbose=False):
        super(SCSADMM, self).__init__(max_iters, atol, do_all_iters,
            initial_iterate=None, verbose=verbose)

    def _u(self, problem, uv):
        return np.hstack((problem.p(uv), problem.y(uv), problem.tau(uv)))

    def _v(self, problem, uv):
        return np.hstack((problem.r(uv), problem.s(uv), problem.kappa(uv)))

    def solve(self, problem):
        if not isinstance(problem, SCSProblem):
            raise ValueError('SCSADMM can only solve SCSProblem instances, '
                'but received an instance of %s' % type(problem))
        product_set = problem.sets[0]
        affine_set = problem.sets[1]
            
        # TODO(akshayka) ... everything else (below code is copied from altp)
        # u = np.ones, v = np.ones ...
        # y_k = affine_set.project(x_k)
        # u_k_plus = u_k_plus_tilde - v_k
        # v_k_plus = v_k - u_k_plus_tilde + u_k_plus
        iterate = np.ones(problem.dimension)
        iterates = [iterate]
        residuals = []

        status = Optimizer.Status.INACCURATE
        for i in xrange(self.max_iters):
            if self.verbose:
                print 'iteration %d' % i
            uv_k = iterates[-1]
            residuals.append(self._compute_residual(
                uv_k, product_set.project(uv_k), affine_set.project(uv_k)))
            if self.verbose:
                print '\tresidual: %e' % sum(residuals[-1])
            if self._is_optimal(residuals[-1]):
                status = Optimizer.Status.OPTIMAL
                if not self.do_all_iters:
                    break

            # Parse uv into its components
            u_k = self._u(problem, uv_k)
            v_k = self._v(problem, uv_k)
            u_k_plus_v_k = u_k + v_k
            # Project onto the affine set and parse
            uv_k_tilde = affine_set.project(np.hstack(
                (u_k_plus_v_k, u_k_plus_v_k)))
            u_k_tilde = self._u(problem, uv_k_tilde)    
            v_k_tilde = self._v(problem, uv_k_tilde)
            # Note that we could have used the Moreau decomposition here to
            # save on computation, but computing both cone projections
            # explicitly is easier with my code
            u_k_prime = u_k_tilde - v_k
            v_k_prime = v_k_tilde - u_k
            uv_k_plus = product_set.project(np.hstack((u_k_prime, v_k_prime)))
            iterates.append(uv_k_plus)
        # TODO(akshayka): Consider polishing the result at this step, or even
        # running apop using the final iterate
        return iterates, residuals, status
