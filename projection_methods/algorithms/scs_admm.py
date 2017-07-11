import numpy as np

from projection_methods.algorithms.apop import APOP
from projection_methods.algorithms.optimizer import Optimizer
from projection_methods.problems.problems import SCSProblem

class SCSADMM(Optimizer):
    def __init__(self,
            max_iters=100, atol=10e-8, do_all_iters=False, polish=False,
            verbose=False):
        super(SCSADMM, self).__init__(max_iters, atol, do_all_iters,
            initial_iterate=None, verbose=verbose)
        self.polish = polish

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
            
        iterate = np.ones(problem.dimension)
        iterates = [iterate]
        residuals = []
        info = []

        status = Optimizer.Status.INACCURATE
        for i in xrange(self.max_iters):
            if self.verbose:
                print 'iteration %d' % i
            uv_k = iterates[-1]
            residuals.append(self._compute_residual(
                uv_k, product_set.project(uv_k), affine_set.project(uv_k)))
            if self.verbose:
                r = residuals[-1]
                print '\tresidual: %e' % sum(r)
                print '\t\tproduct: %e' % r[0]
                print '\t\taffine: %e' % r[1]
            if self._is_optimal(residuals[-1]):
                status = Optimizer.Status.OPTIMAL
                if not self.do_all_iters:
                    break

            # Parse uv into its components
            u_k = self._u(problem, uv_k)
            v_k = self._v(problem, uv_k)
            u_k_plus_v_k = u_k + v_k
            # Project onto the affine set and parse
            uv_k_tilde, h_a = affine_set.query(np.hstack(
                (u_k_plus_v_k, u_k_plus_v_k)))
            u_k_tilde = self._u(problem, uv_k_tilde)    
            v_k_tilde = self._v(problem, uv_k_tilde)
            # Note that we could have used the Moreau decomposition here to
            # save on computation, but computing both cone projections
            # explicitly is easier with my code
            u_k_prime = u_k_tilde - v_k
            v_k_prime = v_k_tilde - u_k
            uv_k_plus, h_p = product_set.query(np.hstack(
                (u_k_prime, v_k_prime)))
            iterates.append(uv_k_plus)
            info.extend(h_a + h_p)
        # TODO(akshayka): Consider polishing the result at this step, or even
        # running apop using the final iterate
        if self.polish:
            polisher = APOP(max_iters=100, atol=self.atol, do_all_iters=True,
                initial_iterate=iterates[-1],
                average=False, info=info, verbose=True)
            it, res, status = polisher.solve(problem)
            iterates.extend(it)
            residuals.extend(res)
        return iterates, residuals, status
