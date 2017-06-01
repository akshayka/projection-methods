import cvxpy

from projection_methods.oracles.convex_set import ConvexSet


class SOC(ConvexSet):
    """An oracle for second order cones

    Defines an oracle for the second order cone, i.e.,
        \{ (y, t) | || ||y||_2 \leq t \}
    """
    def __init__(self, x):
        """
        Args:
            x (cvxpy.Variable): a symbolic representation of
                members of the set
        """
        constr = [cvxpy.norm(x[:-1], 2) <= x[-1]]
        super(SOC, self).__init__(x, constr)
