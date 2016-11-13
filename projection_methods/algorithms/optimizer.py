import abc

class Optimizer(object):
    def __init__(self, max_iters=100, eps=10e-5):
        self._max_iters = max_iters
        self._eps = eps


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
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, eps):
        if eps < 0:
            raise ValueError('eps must be >= 0')
        self._eps = eps
