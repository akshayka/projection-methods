import numpy as np

from projection_methods.oracles.oracle import Oracle
from projection_methods.projectables.polyhedron import Polyhedron
from projection_methods.projectables.projectable import Projectable
import projection_methods.oracles.utils as utils


class ConvexOuter(object):
    POLYHEDRAL, EXACT, EMPTY = range(3)

class ConvexSet(Projectable, Oracle):
    """A projectable oracle for convex sets"""
    def __init__(self, x, constr=[], t=ConvexOuter.EXACT):
        """
        Args:
            x (cvxpy.Variable): a symbolic representation of
                members of the set
            constr (list of cvxpy.Expression): constraints to
                impose on members of set
        """
        self._info= []
        super(ConvexSet, self).__init__(x, constr)


    def query(self, x_0):
        """Projects x_0 onto set, constructing a cutting plane
        
        Computes the projection of x_0 onto the set, and
        adds the containing halfspace defined by the projection
        to the oracle's polyhedral approximation

        Args:
            x_0 (array-like): query point
        Returns:
            array-like: the projection of x_0 onto the set
            list of Halfspace: a halfspace containing the set, defined
                by the supporting hyperplane at the projection
                of x_0 onto the set
        """
        x_star = self.project(x_0)
        if np.array_equal(x_star, x_0):
            return x_0, []
        info = []
        h = utils.containing_halfspace(x_0, x_star, self._x)
        if h is not None:
            self._info.append(h)
            info.append(h)
        return x_star, info


    def outer(self, kind=ConvexOuter.POLYHEDRAL):
        """Returns outer approximation of set

        Args:
            kind: a member of ConvexOuter
        Returns:
            Projectable: an outer approximation of the set
                that implements Projectable
        """
        if kind == ConvexOuter.POLYHEDRAL:
            return Polyhedron(self._x, self._info)
        elif kind == ConvexOuter.EXACT:
            return self
        elif kind == ConvexOuter.EMPTY:
            return Polyhedron(self._x, information=[])
        else:
            raise(ValueError, "Unknown kind " + str(kind))

    def residual(self, x_0):
        return np.linalg.norm(x_0 - self.project(x_0), 2) ** 2

    def residual_str(self, x_0):
        string = self.__repr__()
        string += '\t res: %e' % self.residual(x_0)
        return string
