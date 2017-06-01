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
    class Eviction(object):
        """Eviction policies for halfspaces/hyperplanes

        LRA: Evict least recently added 
        RANDOM: Evict random candidate
        MRA: Evict most recently added
        """
        LRA, RANDOM, MRA = range(3)


    def __init__(self, x, information=[],
            max_hyperplanes=float("inf"),
            max_halfspaces=float("inf"),
            eviction_policy=Eviction.LRA):
        """
        Args:
        x (cvxpy.Variable): a symbolic representation of
            members of the set
        information (list of Halfspace and/or Hyperplane): halfspaces and
            hyperplanes defining polyhedron
        max_hyperplanes: maximum hyperplane capacity
        max_halfspaces: maximum halfspace capacity
        eviction_policy: a member of Eviction
        """
        self._hyperplanes = []
        self._halfspaces = []
        self.max_hyperplanes = max_hyperplanes
        self.max_halfspaces = max_halfspaces
        self.add(information)
        self.eviction_policy = eviction_policy
        constr = [c for h in self._hyperplanes for c in h._constr]
        constr += [c for h in self._halfspaces for c in h._constr]
        super(Polyhedron, self).__init__(x, constr)


    def halfspaces(self): return self._halfspaces
    def hyperplanes(self): return self._hyperplanes

    def add(self, information):
        """Add hyperplanes and halfspaces to polyhedron

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
                self._add_hyperplane(info)
            elif type(info) == Halfspace:
                self._add_halfspace(info)
            else:
                raise ValueError, "Only Halfspaces or Hyperplanes can be added"


    def _add_hyperplane(self, hyperplane):
        if len(self._hyperplanes) == self.max_hyperplanes:
            # TODO(akshayka): implement eviction
            pass
        self._hyperplanes.append(hyperplane)


    def _add_halfspace(self, halfspace):
        if len(self._halfspaces) == self.max_halfspaces:
            # TODO(akshayka): implement eviction
            pass
        self._halfspaces.append(halfspace)
