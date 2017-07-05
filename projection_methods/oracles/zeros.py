import numpy as np
import scipy.sparse

from projection_methods.oracles.cone import Cone
from projection_methods.projectables.hyperplane import Hyperplane


class Zeros(Cone):
    """A (trivial) oracle for \{0\}^n"""
    def __init__(self, x):
        """
        Args:
            x (cvxpy.Variable): a symbolic representation of
                members of the set
        """
        constr = [x == 0]
        super(Zeros, self).__init__(x, constr)

    def contains(self, x_0, atol=1e-8):
        return not np.any(np.absolute(x_0) > atol)

    def project(self, x_0):
        return x_0 if self.contains(x_0) else np.zeros(x_0.shape)

    def dual(self, x):
        return Reals(x)

    def query(self, x_0):
        x_star = self.project(x_0) 
        h = []
        # This is an abuse of the word hyperplane; this function
        # actually returns a set of hyperplanes that exactly identifies
        # the zero set. If added to an outer approximation, the
        # interpretation is that we are doing a presolve by forcing
        # x to be 0.
        # TODO(akshayka): It is not clear to me, at all, whether it is a good
        # idea to return such a construct.
        A = scipy.sparse.eye(x_0.shape[0])
        h = [Hyperplane(self._x, A, 0)]
        return x_star, h


class Reals(Cone):
    """A (trivial) oracle for R^n"""
    def __init__(self, x):
        """
        Args:
            x (cvxpy.Variable): a symbolic representation of
                members of the set
        """
        constr = []
        super(Reals, self).__init__(x, constr)

    def contains(self, x_0, atol=1e-6):
        return True

    def project(self, x_0):
        return x_0

    def dual(self, x):
        return Zeros(x)

    def query(self, x_0):
        return x_0, []
