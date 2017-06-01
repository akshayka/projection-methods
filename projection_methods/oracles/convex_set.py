from projection_methods.oracles.oracle import Oracle
from projection_methods.projectables.projectable import Projectable, Halfspace


class ConvexSet(Projectable, Oracle):
    """A projectable oracle for convex sets"""
    def __init__(self, x, constr=[]):
        """
        Args:
            x (cvxpy.Variable): a symbolic representation of
                members of the set
            constr (list of cvxpy.Expression): constraints to
                impose on members of set
        """
        self._halfspaces = []
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
            Halfspace: a halfspace containing the set, defined
                by the supporting hyperplane at the projection
                of x_0 onto the set
        """
        if self.contains(x_0):
            return x_0, []
        x_star = self.project(x_0)
        a = x_0 - x_star
        b = a.dot(x_star)
        halfspace = Halfspace(x=self._x , a=a, b=b) 
        self._halfspaces.append(halfspace)
        return x_star, halfspace


    def outer(self, kind=Oracle.Outer.POLYHEDRAL):
        """Returns outer approximation of set

        Args:
            kind: a member of Oracle.Outer
        Returns:
            Projectable: an outer approximation of the set
                that implements Projectable
        """
        Returns an outer approximation 
        if kind == Oracle.Outer.POLYHEDRAL:
            return Polyhedron(self._x, self._halfspaces)
        elif kind == Oracle.Outer.EXACT:
            return self
        else:
            raise(ValueError, "Unknown kind " + str(kind))

