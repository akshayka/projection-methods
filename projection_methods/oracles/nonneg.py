from projection_methods.oracles.cone import Cone

import numpy as np


# TODO(akshayka): should query return a unique halfspace each time?
# (i.e., x_1 >= 0, x_2 >= 0, ... and so on?) This doesn't seem to help
# much
class NonNeg(Cone):
    """A (trivial) oracle for the nonnegative orthant."""
    def __init__(self, x):
        """
        Args:
            x (cvxpy.Variable): a symbolic representation of
                members of the set
        """
        constr = [x >= 0]
        super(NonNeg, self).__init__(x, constr)
        self._i = 0

    def contains(self, x_0, atol=1e-6):
        return (x_0 >= 0).all()

    def project(self, x_0):
        return np.maximum(x_0, 0)

    def dual(self, x):
        return NonNeg(x)
