from projection_methods.projectables.halfspace import Halfspace
from projection_methods.projectables.hyperplane import Hyperplane
from projection_methods.projectables.projectable import Projectable


class Polyhedron(Projectable):
    """A projectable polyhedron

    Defines a polyhedron via a set of hyperplanes and halfspaces. An
    unconventional Projectable in that it implements an optional maximum
    size (defined as a maximum number of halfspaces and a maximum number of
    hyperplanes) and implements policies by which halfspaces and hyperplanes
    are evicted when maximum capacity is met.

    Attributes:
        max_hyperplanes: maximum number of hyperplanes (default infinite)
        max_halfspaces: maximum number of halfspaces (default infinite)
        eviction_policy: policy by which to evict halfspaces
    """
    def __init__(self, x, information=[]):
        """
        Args:
        x (cvxpy.Variable): a symbolic representation of
            members of the set
        information (list of Halfspace and/or Hyperplane): halfspaces and
            hyperplanes defining polyhedron
        """
        self._hyperplanes = []
        self._halfspaces = []
        self._constr = []
        self.add(information)
        super(Polyhedron, self).__init__(x, self._constr)


    def halfspaces(self): return self._halfspaces
    def hyperplanes(self): return self._hyperplanes

    def add(self, information):
        """Adds hyperplanes and halfspaces to polyhedron

        Args:
            information (list of Hyperplane and/or Halfspace): halfspaces and
                hyperplanes to add to polyhedron
        Raises:
            ValueError if information contains an object that is not a
                Hyperplane or a Halfspace
        """
        if type(information) is not list:
            information = [information]
        for info in information:
            if type(info) == Hyperplane:
                self._hyperplanes.append(info)
                self._constr += [c for c in info._constr]
            elif type(info) == Halfspace:
                self._halfspaces.append(info)
                self._constr += [c for c in info._constr]
            else:
                raise ValueError, "Only Halfspaces or Hyperplanes can be added"
