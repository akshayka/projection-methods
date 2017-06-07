import numpy as np

from projection_methods.oracles.convex_set import ConvexSet


class CartesianProduct(ConvexSet):
    """An oracle for cartesion products of convex sets

    Defines an oracle for the Cartesian product of an arbitrary
    number of convex sets :math:`C_1, C_2, \ldots, C_n`, i.e.,
        S := C_1 \times C_2 \times \ldots \times C_n

    Attributes:
        sets (list of ConvexSet): sets C_1, \ldots, C_n
        slices (list of slice): if the Cartesian product lies in
            R^d, then slices[i] is the slice of [0 ... d-1] that
            corresponds to C_i. For example, if n == 2 and
                C_1 \subseteq R^10,
                C_2 \subseteq R^20 (so d == 30),
            then
                slices[0] == slice(0, 10) and
                slices[1] == slice(10, 30).
    """
    def __init__(self, sets, slices):
        """
        Args:
            sets (list of ConvexSet): as per attribute
            slices (list of slice): as per at atribute
        """
        self.sets = sets
        self.slices = slices
        x = self.sets[0]._var
        constr = [c for s in sets for c in s._constr]
        super(CartesianProduct, self).__init__(x, constr)


    def project(self, x_0):
        x_star = np.zeros(x_0.shape)
        for s, slx in zip(self.sets, self.slices):
            x_star[slx] = s.project(x_0[slx])
        return x_star
        
    
    def query(self, x_0):
        """As ConvexSet.query, but returns a list of Halfspaces/Hyperplanes

        Computes a halfspace/hyperplane for each convex set C_i in the
        Cartesian product, instead of a single halfspace/hyperplanes for the
        entire set

        Args:
            x_0 (array-like): query point
        Returns:
            array-like: the projection of x_0 onto the set
            list of Halfspace and/or Hyperplane: a list of
                halfspspaces/hyperplanes such that, for every point x in the
                Cartesian product, the i-th slice of x lies in the i-th
                halfspace/hyperplane
        """
        x_star = np.zeros(x_0.shape)
        hyper_half= []
        for s, slx in zip(self.sets, self.slices):
            x_s, h_s = s.query(x_0[slx])
            x_star[slx] = x_s
            hyper_half.extend(h_s)
        # Note the lazy English below; self._halfspaces here may very well
        # contain hyperplanes.
        self._halfspaces.extend(hyper_half)
        return x_star, hyper_half
