import copy
import numpy as np

from projection_methods.oracles.cone import Cone
import projection_methods.oracles.utils as utils


class CartesianProduct(Cone):
    """An oracle for cartesion products of cones

    Defines an oracle for the Cartesian product of an arbitrary
    number of cones C_1, C_2, \ldots, C_n, i.e.,
        S := C_1 \times C_2 \times \ldots \times C_n

    Attributes:
        sets (list of Cone): sets C_1, \ldots, C_n
        slices (list of slice): if the Cartesian product lies in
            R^d, then slices[i] is the slice of [0 ... d-1] that
            corresponds to C_i. For example, if n == 2 and
                C_1 \subseteq R^10,
                C_2 \subseteq R^20 (so d == 30),
            then
                slices[0] == slice(0, 10) and
                slices[1] == slice(10, 30).
    """
    def __init__(self, x, sets, slices):
        """
        Args:
            sets (list of Cone): as per attribute
            slices (list of slice): as per atribute
        """
        for s in sets:
            assert isinstance(s, Cone)

        self.sets = sets
        self.slices = slices
        constr = [c for s in sets for c in s._constr]
        super(CartesianProduct, self).__init__(x, constr)
        self._shape = reduce(
            lambda x, y: tuple(one + two for one, two in zip(x, y)),
            [s._shape for s in sets])
        assert self._shape == np.prod(x.size)


    def project(self, x_0):
        assert self._shape == x_0.shape, \
            'cone shape (%s) != x_0 shape (%s)' % (
            str(self._shape), str(x_0.shape))
        x_star = np.zeros(x_0.shape)
        for s, slx in zip(self.sets, self.slices):
            x_star[slx] = s.project(x_0[slx])
        return x_star


    def dual(self, x):
        # TODO(akshayka): assert that x is of the correct size
        cones = []
        for s, slx in zip(self.sets, self.slices):
            cones.append(s.dual(x[slx]))
        return CartesianProduct(x, cones, copy.copy(self.slices))
            
    
    def query(self, x_0, granular=True):
        """As ConvexSet.query, but returns a list of Halfspaces/Hyperplanes

        Computes a halfspace/hyperplane for each cone C_i in the
        Cartesian product, instead of a single halfspace/hyperplane for the
        entire set

        TODO(akshayka): Add the option to return a single halfspace for the
                        entire cone.

        Args:
            x_0 (array-like): query point
        Returns:
            array-like: the projection of x_0 onto the set
            list of Halfspace and/or Hyperplane: a list of
                halfspspaces/hyperplanes such that, for every point x in the
                Cartesian product, the i-th slice of x lies in the i-th
                halfspace/hyperplane
        """
        info = []
        if granular:
            x_star = np.zeros(x_0.shape)
            for s, slx in zip(self.sets, self.slices):
                x_s, h_s = s.query(x_0[slx])
                assert x_s.shape == x_0[slx].shape
                x_star[slx] = x_s
                info.extend(h_s)
        else:
            x_star = self.project(x_0)
            if not np.array_equal(x_star, x_0):
                h = utils.containing_halfspace(x_0, x_star, self._x)
                if h != None:
                    info.append(h)
        # Note the lazy English below; self._halfspaces here may very well
        # contain hyperplanes if info contains hyperplanes.
        self._info.extend(info)
        return x_star, info


    def residual(self, x_0):
        """Compute distance from x_0 to the cartesian product.

        Args:
            x_0 (array-like): query point
        Returns:
            list : list of residuals, possibly nested
        """
        return [s.residual(x_0[slx]) for s, slx in zip(self.sets, self.slices)]


    def residual_str(self, x_0):
        string = '------- Cone Residuals -------\n'
        for i, tup in enumerate(zip(self.sets, self.slices)):
            s, slx = tup
            string += '%d. %s\n' % (i+1, s.residual_str(x_0[slx]))
        return string


    def __repr__(self):
        string = type(self).__name__ + "\n"
        string += 'Number of cones: %s\n' % str(len(self.sets))
        string += 'Dimension: %s\n' % str(self._shape)
        string += '------- Cones -------\n'
        for i, c in enumerate(self.sets):
            string += '%d. %s' % (i+1, c.__repr__())
        return string
