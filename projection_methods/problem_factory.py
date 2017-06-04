import cvxpy as cvx
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from projection_methods.oracles.affine_set import AffineSet
from projection_methods.problems import FeasibilityProblem


def scs_like_problem(cones, u_shape, v_shape):
    """Generates a random feasibility problem in the style of SCS

    Generates a random feasibility problem in the style of the splitting
    conic solver (SCS; see http://web.stanford.edu/~boyd/papers/scs.html).

        find (u, v)
        s.t. [Q, 0; 0, -I] * [u; v] = 0
             (u, v) \in C \times C^*, C = R^n \times K^* \times R_+,
                                      C^* = {0}^n \times K \times R_+  

    Note that this problem _always_ admits zero as a solution, so in general
    the solution to a problem returned by this function is not unique. The
    returned problem is always feasible.

    Args:
        cones (CartesianProduct): The CartesianProduct C \times C^* (see above)
        u_shape (tuple): the dimension of the vector u
        v_shape (tuple): the dimension of the vector v
    Returns:
        FeasibilityProblem: an SCS-like problem
    """
    # TODO(akshayka): Implement this.
    pass
        
       
def convex_affine_problem(convex_set, shape, density=0.01):
    """Generates a random feasibility problem w.r.t. a convex and affine set

    Generates a feasibility problem in which the first set is some convex
    set and the second is an affine set. In particular, the returned problem is
    of the form

        find x
        s.t. Ax = b, A full-rank
             x \in C
    

    If the convex set is in fact a cross product of some number of cones, then
    the generated problem roughly resembles those solved by SCS.  The returned
    problem is always feasible.

    Args:
        convex_set (ConvexSet): a ConvexSet object; one of two sets w.r.t.
            which the feasibility problem will be defined
        shape (tuple): the shape of the data matrix A to be generated; A will
            be full-rank
        density (float): the density of A; a number in (0, 1]

    Returns:
        FeasbilityProblem: a FeasibilityProblem where sets[0] := convex_set
            and sets[1] := the generated affine set (an instance of AffineSet)
    """
    
    x_0 = np.random.uniform(-1, 1, size=shape[0])
    x_opt = convex_set.project(x_0)

    A = scipy.sparse.rand(m=shape[0], n=shape[1], density=density, format='csc')
    b = A.dot(x_opt)
        
    return FeasibilityProblem([convex_set, AffineSet(convex_set._x, A, b)],
        x_opt)
