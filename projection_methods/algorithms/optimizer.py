import abc

import numpy as np

class Optimizer(object):
    class Status(object):
        OPTIMAL, INACCURATE, INFEASIBLE = range(3)


    def __init__(self, max_iters=100, atol=10e-8, do_all_iters=False,
        initial_iterate=None, verbose=False):
        self._max_iters = max_iters
        self._atol = atol
        self.do_all_iters = do_all_iters
        self._initial_iterate = initial_iterate
        self.verbose = verbose


    @abc.abstractmethod
    def solve(self, problem):
        pass


    @property
    def max_iters(self):
        return self._max_iters

    @max_iters.setter
    def max_iters(self, max_iters):
        if max_iters <= 0 or not isinstance(max_iters, int):
            raise ValueError('max iters must be > 0')
        self._max_iters = max_iters

    @property
    def atol(self):
        return self._atol

    @atol.setter
    def atol(self, atol):
        if atol < 0:
            raise ValueError('atol must be >= 0')
        self._atol = atol

    def _compute_residual(self, x_k, y_k, z_k):
        """Returns tuple (dist from left set squared,
        dist from right set squared)"""
        # TODO(akshayka): should these distances be squared? does it matter?
        return (
            np.linalg.norm(x_k - y_k, 2)**2,
            np.linalg.norm(x_k - z_k, 2)**2)

    def _is_optimal(self, r_k):
        return reduce(lambda x, y: x and y, [r <= self.atol for r in r_k])
