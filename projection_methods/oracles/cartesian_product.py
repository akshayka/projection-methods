from oracles.convex_set import ConvexSet


class CartesianProduct(ConvexSet):
    """An oracle for cartesion products of convex sets

    Defines an oracle for the Cartesian product of an arbitrary
    number of convex sets :math:`C_1, C_2, \ldots, C_n`, i.e.,
        S := C_1 \times C_2 \times \ldots \times C_n

    Attributes:
        sets (list of ConvexSet): sets C_1, \ldots, C_n
        dims (list of slice): if the Cartesian product lies in
            R^d, then dims[i] is the slice of [0 ... d-1] that
            corresponds to C_i. For example, if n == 2 and
                C_1 \subseteq R^10,
                C_2 \subseteq R^20 (so d == 30),
            then
                dims[0] == slice(0, 10) and
                dims[1] == slice(10, 30).
    """
    def __init__(sets, dims):
        """
        Args:
            sets (list of ConvexSet): as per attribute
            dims (list of slice): as per attribute
        """
        self.sets = sets
        self.dims = dims
        x = self.sets[0]._x
        constr = [c for s in sets for c in s._constr]
        super(CartesianProduct, self).__init__(x, constr)


    def query(x_0):
        """As ConvexSet.query, but returns a list of Halfspaces

        Computes a halfspace for each convex set C_i in the Cartesian product,
        instead of a single halfspace for the entire set

        Args:
            x_0 (array-like): query point
        Returns:
            array-like: the projection of x_0 onto the set
            list of Halfspace: a list of halfspspaces such that, for every
                point x in the Cartesian product, the i-th slice of x lies in
                the i-th halfspace
        """
        x_star = np.zeros(x_0.shape)
        halfspaces = []
        for s, dim in zip(self.sets, self.dims):
            x_s, h_s = s.query(x_0[dim])
            x_star[dim] = x_s
            halfspaces.append(h_s)
        self._halfspaces.extend(halfspaces)
        return x_star, halfspaces
