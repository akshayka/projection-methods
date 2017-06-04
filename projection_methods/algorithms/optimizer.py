import abc

class Optimizer(object):
    class Status(object):
        OPTIMAL, INACCURATE, INFEASIBLE = range(3)


    def __init__(self, max_iters=100, atol=10e-5, initial_iterate=None):
        self._max_iters = max_iters
        self._atol = atol
        self._initial_iterate = initial_iterate


    @abc.abstractmethod
    def solve(self, problem, options={}):
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
