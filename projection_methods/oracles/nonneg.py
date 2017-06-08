from projection_methods.oracles.convex_set import ConvexSet

import numpy as np


# TODO(akshayka): should query return a unique halfspace each time?
# (i.e., x_1 >= 0, x_2 >= 0, ... and so on?)
class NonNeg(ConvexSet):
    """A (trivial) oracle for the nonnegative orthant."""
    def __init__(self, x):
        """
        Args:
            x (cvxpy.Variable): a symbolic representation of
                members of the set
        """
        constr = [x >= 0]
        super(NonNeg, self).__init__(x, constr)

    def contains(self, x_0, atol=1e-6):
        return (x_0 >= 0).all()

    def project(self, x_0):
        return np.maximum(x_0, 0)
