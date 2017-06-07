import cvxpy
import numpy as np

from projection_methods.oracles.convex_set import ConvexSet


class SOC(ConvexSet):
    """An oracle for second order cones

    Defines an oracle for the second order cone, i.e.,
        \{ (z, t) | || ||z||_2 \leq t \}
    """
    def __init__(self, x):
        """
        Args:
            x (cvxpy.Variable): a symbolic representation of
                members of the set
        """
        constr = [cvxpy.norm(x[:-1], 2) <= x[-1]]
        super(SOC, self).__init__(x, constr)


    def _contains(self, norm_z, t, atol=1e-8):
        return norm_z <= t or np.isclose(norm_z, t, atol=atol)

    def contains(self, x_0, atol=1e-6):
        z = x_0[:-1]
        t = x_0[-1]
        return self._contains(np.linalg.norm(z, 2), t)

    def project(self, x_0):
        z = x_0[:-1]
        t = x_0[-1]
        norm_z = np.linalg.norm(z, 2)

        # As given in [BV04, Chapter 8.Exercises]
        if norm_z <= -t:
            # this certainly will never happen when t > 0
            return np.zeros(np.shape(x_0))
        elif self._contains(norm_z, t):
            return x_0
        else:
            return 0.5 * (1 + t/norm_z) * np.append(z, norm_z)
