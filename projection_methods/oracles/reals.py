from projection_methods.oracles.cone import Cone


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
