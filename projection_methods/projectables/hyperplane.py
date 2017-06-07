import numpy as np

from projection_methods.projectables.projectable import Projectable


class Hyperplane(Projectable):
    """A projectable hyperplane

     Defines a hyperplane of the form
        \{ x | <a, x> = b \}.
     """
    def __init__(self, x, a, b):
        """
        Args:
            x (cvxpy.Variable): a symbolic representation of
                members of the set
            a (array-like): normal vector
            b (array-like): offset vector
        """
        self.a = a
        self.b = b
        constr = [a.T * x == b]
        super(Hyperplane, self).__init__(x, constr)


    def contains(self, x_0, atol=1e-4):
        """Return True if x_0 in halfspace, False otherwise"""
        return np.allclose(self.a.dot(x_0), self.b, atol=atol)

    def project(self, x_0):
        if self.contains(x_0):
            return x_0
        else:
            return super(Hyperplane, self).project(x_0)


    def __repr__(self):
        string = type(self).__name__ + "\n"
        string += "a: " + str(self.a) + "\n"
        string += "b: " + str(self.b)
        return string


