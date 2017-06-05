from projection_methods.oracles.convex_set import ConvexSet


class Zeros(ConvexSet):
    """A (trivial) oracle for \{0\}^n"""
    def __init__(self, x):
        """
        Args:
            x (cvxpy.Variable): a symbolic representation of
                members of the set
        """
        constr = [x == 0]
        super(Zeros, self).__init__(x, constr)

    def contains(self, x_0, atol=1e-6):
        return not np.any(np.absolute(my_array) > atol)

    def project(self, x_0):
        return x_0 if self.contains(x_0) else np.zeros(x_0.shape)

    def query(self, x_0):
        """NB: The  halfspace does not give information, so not returned."""
        x_star = self.project(x_0) 
        return x_star, []
