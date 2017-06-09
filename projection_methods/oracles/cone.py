import abc

from projection_methods.oracles.convex_set import ConvexSet


class Cone(ConvexSet):
    @abc.abstractmethod
    def dual(self, x):
        """Return the dual cone of this cone, which is itself a Cone instance.

        Args:
            x (cvxpy.Variable): a symbolic representation of
                members of the set

        Returns:
           Cone: the dual cone of this cone, in which x is made to reside.
        """
        pass
