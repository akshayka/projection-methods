import numpy as np

from projection_methods.oracles.oracle import Oracle
from projection_methods.projectables.hyperplane import Hyperplane
from projection_methods.projectables.halfspace import Halfspace
from projection_methods.projectables.polyhedron import Polyhedron
from projection_methods.projectables.projectable import Projectable


class PolyOuter(object):
    """Management policies for halfspaces/hyperplanes

    EXACT: Renders the oracle a dummy wrapper for its
        polyhedron
    ELRA: Evict least recently added
    ERANDOM: Evict random candidate
    SUBSAMPLE: Expose a random subsample of the hyperplanes/halfspaces
        such that, in expectation, max_hyperplanes / max_halfspace
        hyperplanes/halfspaces are exposed in each call to outer

    TODO(akshayka):
        add EMRA: evict most recently added (weird idea)
    """
    EXACT, ELRA, ERANDOM, ERESET, SUBSAMPLE = range(5)
    POLICIES = frozenset([EXACT, ELRA, ERANDOM, ERESET, SUBSAMPLE])
    EVICTIONS = frozenset([ELRA, ERANDOM, ERESET])


class DynamicPolyhedron(Oracle):
    """Dynamic manager for a Polyhedron

    A DynamicPolyhedron mediates access to a Polyhedron by exposing a finite
    number of the hyperplanes and halfspaces defining it; the exposed
    hyperplanes and halfspaces are an outer approximation of the polyhedron.
    Clients can select from a number of policies that determine the
    construction of the outer approximation.

    Unlike ConvexSet, DynamicPolyhedron does _not_ implement the Projectable
    interface; `query` interrogates the outer approximation of the managed
    polyhedron instead of the complete description of the set.

    Attributes:
        max_hyperplanes: maximum number of hyperplanes (default infinite)
        max_halfspaces: maximum number of halfspaces (default infinite)
        policy: policy by which to construct outer approximations
    """
    def __init__(self, polyhedron, max_hyperplanes=float("inf"),
            max_halfspaces=float("inf"), policy=PolyOuter.EXACT):
        """
        Args:
        x (cvxpy.Variable): a symbolic representation of
            members of the set
        information (list of Halfspace and/or Hyperplane): halfspaces and
            hyperplanes defining polyhedron
        max_hyperplanes: maximum hyperplanes exposed
        max_halfspaces: maximum halfspaces exposed
        policy: a member of PolyOuter
        """
        self._polyhedron = polyhedron
        self._outer_hyperplanes = []
        self._outer_halfspaces = []
        self.max_hyperplanes = max_hyperplanes
        self.max_halfspaces = max_halfspaces

        assert policy in PolyOuter.POLICIES
        if policy == PolyOuter.EXACT:
            assert self.max_hyperplanes == float("inf")
            assert self.max_halfspaces == float("inf")
        self.policy = policy


    def add(self, information):
        """Adds hyperplanes and halfspaces to outer set, polyhedron

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
                if self.max_hyperplanes > 0:
                    self._add_hyperplane(info)
            elif type(info) == Halfspace:
                if self.max_halfspaces > 0:
                    self._add_halfspace(info)
            else:
                raise ValueError, "Only Halfspaces or Hyperplanes can be added"


    def query(self, x_0):
        """Query the outer approximation

        Args:
            x_0 (array-like): query point

        Returns:
            array-like: the projection of x_0 onto outer(); note that
                x_0 \in managed polyhedron \implies x_star == x_0, but
                the converse is not true.
        """
        x_star, h = self.outer().query(x_0)
        return x_star, h


    def outer(self):
        """Returns polyhedron defined by the current outer approximation

        The outer approximation of hyperplanes and halfspaces depends upon the
        DynamicPolyhedron's policy and its capacities for hyperplanes and
        halfspaces. This method computes the approximation and wraps it in a
        Polyhedron object, which is returned to the client. Note that the
        returned polyhedron is not necessarily the same object as the
        DynamicPolyhedron's attribute `polyhedron`.

        Returns:
            Polyhedron: the outer approximation
        """
        if (self.policy == PolyOuter.EXACT or (
            len(self._polyhedron.hyperplanes()) <= self.max_hyperplanes and
            len(self._polyhedron.halfspaces()) <= self.max_halfspaces)):
            return self._polyhedron
        else:
            if self.policy == PolyOuter.SUBSAMPLE:
                if len(self._polyhedron.hyperplanes()) > self.max_hyperplanes:
                    self._outer_hyperplanes = list(np.random.choice(
                        self._polyhedron.hyperplanes(),
                        size=self.max_hyperplanes, replace=False))
                if len(self._polyhedron.halfspaces()) > self.max_halfspaces:
                    self._outer_halfspaces = list(np.random.choice(
                        self._polyhedron.halfspaces(),
                        size=self.max_halfspaces, replace=False))
            # eviction policies maintain outer as information is added
            return Polyhedron(self._polyhedron._x,
                self._outer_hyperplanes + self._outer_halfspaces)


    def _evict(self, items, max_len):
        if self.policy == PolyOuter.ELRA:
            return items[len(items)-max_len+1:]
        elif self.policy == PolyOuter.ERANDOM:
            evict_index = random.randint(0, max_len - 1)
            items[0:evict_index] + items[evict_index + 1:]
        elif self.policy == PolyOuter.ERESET:
            return []
        else:
            raise RuntimeError('_evict invoked, but policy is not an eviction')


    def _add_hyperplane(self, hyperplane):
        if (len(self._outer_hyperplanes) >= self.max_hyperplanes and
                self.policy in PolyOuter.EVICTIONS):
            self._outer_hyperplanes = self._evict(self._outer_hyperplanes,
                self.max_hyperplanes)
        self._outer_hyperplanes.append(hyperplane)
        self._polyhedron.add(hyperplane)


    def _add_halfspace(self, halfspace):
        if (len(self._outer_halfspaces) >= self.max_halfspaces and
                self.policy in PolyOuter.EVICTIONS):
            self._outer_halfspaces = self._evict(self._outer_halfspaces,
                self.max_halfspaces)
        self._outer_halfspaces.append(halfspace)
        self._polyhedron.add(halfspace)
