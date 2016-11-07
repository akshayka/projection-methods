import abc

class Optimizer(object):


    @abc.abstractmethod
    def solve(self, problem, options={}):
        pass
