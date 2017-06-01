import numpy as np

import projection_methods.algorithms.utils import project


class Projectable(object):
    def __init__(self, x, constr=[]):
        """
        x (cvxpy.Variable): a symbolic representation of
            members of the set
        constr (list of cvxpy.Expression): constraints to impose on members
            of set
        """
        self._x = x
        self._constr = constr
        variables = set([v for c in constr for v in c.variables()])
        assert len(variables) <= 1, ("ConvexSet expects at most one variable "
            "to be constrained among its constraints, "
            "received %d" % len(variables))
        if len(variables) == 1:
            constrained = variables.pop()
            assert constrained is self._x, ("Constrained "
                "variable must be exactly the variable supplied to __init__; "
                "constrained name: %s, supplied name: %s" %
                (constrained._name, self._x._name))


    def project(self, x_0):
        """Project x_0 onto set

        Args:
            x_0 (array-like): point to project
        Returns:
            array-like: projection of x_0 onto set
        """
        x_star = project(x_0, self._constr, self._x)
        return x_star


    def contains(self, x_0):
        """Returns True if x_0 in set, False otherwise"""
        return all(np.isclose(self.project(x_0), x_0, atol=1e-3))


    def __repr__(self):
        string = type(self).__name__ + "\n"
        for c in self._constr:
            string += c.__str__() + "; "
        return string
