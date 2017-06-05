import numpy as np

from projection_methods.oracles.convex_set import ConvexSet
from projection_methods.projectables.hyperplane import Hyperplane

class AffineSet(ConvexSet):
    """An oracle for affine sets
    
    Defines an oracle for affine sets of the form
        \{ x | Ax = b \},
    parametrized by A, b

    Attributes:
        x (cvxpy.Variable): a symbolic representation of
            members of the set
        A (numpy.ndarray or scipy.sparse matrix): a matrix
        b (numpy.ndarray): a target vector
    """
    def __init__(self, x, A, b):
        """
        Args:
            x: see 
            A (numpy.ndarray): a matrix
            b (numpy.ndarray): a target vector
        """
        constr = [A * x == b]
        self.A = A
        self.b = b
        self._hyperplanes = []
        super(AffineSet, self).__init__(x, constr)
        self._shortcut = True


    def contains(self, x_0, atol=1e-4):
        """Return True if x_0 in affine set, False otherwise"""
        return np.allclose(self.A.dot(x_0), self.b, atol=atol)

    def query(self, x_0):
        """As ConvexSet.query, but returns a Hyperplane

        Args:
            x_0 (array-like): query point
        Returns:
            array-like: the projection of x_0 onto the set
            list of Hyperplane: a hyperplane of the form <a, x> = b
                in which every point x in the affine set must lie
        """
        x_star = self.project(x_0)

        # a.dot(y - x_star) == 0, for all y in affine set
        # <==> a.dot(y) == a.dot(x_star)
        a = x_0 - x_star
        b = a.dot(x_star)
        hyperplane = Hyperplane(x=self._x , a=a, b=b) 
        self._hyperplanes.append(hyperplane)
        return x_star, [hyperplane]

    def __repr__(self):
        string = type(self).__name__ + '\n'
        string += 'A of shape %s\n' % str(self.A.shape)
        string += 'b of shape %s' % str(self.b.shape)
        return string


