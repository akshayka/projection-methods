import numpy as np

from projection_methods.projectables.halfspace import Halfspace


def containing_halfspace(x_0, x_star, x):
    """Returns a halfspace containing a convex set

    Args:
        x_0 (array-like): query point
        x_star (array-like): projection of x_0 onto a convex set
        x (CVXPY Variable): variable to constrain
    Returns:
        Halfspace: a halfspace of the form a.T(x) <= b,
            where a = x_0 - x_star, b = a.T(x_star)
    """
    # a = x_0 - x_star
    # a.T(y - x_star) <= 0 for all y in C
    # <==> a.T(y) <= a.T(x_star) := b
    a = x_0 - x_star
    b = a.dot(x_star)
    # TODO(akshayka): Return None if a is too close to 0
    return Halfspace(x=x , a=a, b=b) 
