import cvxpy as cvx
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from projection_methods.oracles.affine_set import AffineSet
from projection_methods.oracles.cartesian_product import CartesianProduct
from projection_methods.oracles.nonneg import NonNeg
from projection_methods.oracles.reals import Reals
from projection_methods.oracles.soc import SOC
from projection_methods.oracles.zeros import Zeros
from projection_methods.problems.problems import FeasibilityProblem
from projection_methods.problems.problems import SCSProblem


def random_cone_program(x, m, n, density=0.01):
    """Generates a random second-order cone program in SCS form

    Generates a random feasibility problem in the style of the splitting
    conic solver (SCS; see http://web.stanford.edu/~boyd/papers/scs.html).

    Note that this problem _always_ admits zero as a solution, so in general
    the solution to a problem returned by this function is not unique. The
    returned problem is always feasible.

    Args:
        x (cvxpy.Variable): The variable to constrain (corresponds to (u, v));
            x _must_ be of shape (2 * (m + n + 1), 1).
        m (int): the number of constraints
        n (int): the number of variables; at most m
        density (float): the density of A; a number in (0, 1]
        
    Returns:
        SCSProblem: a random, feasible SCSProblem
    """
    # Construct the cartesian product: (u, v) \in product 'classes', below
    assert m >= n
    dims = [n, m, 1, n, m, 1]
    assert x.size == (sum(dims), 1)
    #          0      1    2       3      4    5
    classes = [Reals, SOC, NonNeg, Zeros, SOC, NonNeg]
    sets = []
    slices = []
    left = 0
    for d, cls in zip(dims, classes):
        right = left + d
        slx = slice(left, right)
        slices.append(slx)
        sets.append(cls(x[slx]))
        left = right
    product_set = CartesianProduct(sets, slices)

    cone = sets[4]
    # TODO(akshayka): Remove this assert when past debugging phase.
    assert isinstance(cone, SOC)
    
    # Construct members of the optimal set, problem data, and the optimal value;
    # recall that (u, v) := (p, y, tau, r, s, kappa). First generate s, by
    # projecting onto the cone, and y, by Moreau.
    z = np.random.randn(m)
    s = cone.project(z)
    y = s - z 

    # Randomly generate the data matrix
    A = scipy.sparse.rand(m=m, n=n, density=density, format='csc')
    A.data = scipy.randn(A.nnz)

    # Generate an optimal point p (called 'x' in the SCS paper), and data b, c.
    p = np.random.randn(n)
    b = A.dot(p) + s
    c = -A.T.dot(y)
        
    # Construct the augmented KKT matrix
    cm = np.matrix(c).T
    bm = np.matrix(b).T
    Q = scipy.sparse.bmat([
        [None,   A.T,  cm  ],
        [-A,     None, bm  ],
        [-cm.T, -bm.T, None]
    ])
    Q_dim = m + n + 1
    assert Q.shape == (Q_dim, Q_dim)

    # Qu = v if and only if [Q, -I] * [u,v].T = 0
    Q_tilde = scipy.sparse.bmat([[Q, -1 * scipy.sparse.eye(Q.shape[0])]])
    assert Q_tilde.shape == (Q_dim, 2 * Q_dim)
    affine_set = AffineSet(x=x, A=Q_tilde, b=np.zeros(Q_tilde.shape[0]))

    return SCSProblem(sets=[product_set, affine_set], x_opt=None,
        Q=Q, A=A, b=b, c=c, p_opt=p)
     
       
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
    # check that the matrix multiplication makes sense
    dimension = convex_set._var.size[0]
    assert dimension == shape[1]
    x_0 = np.random.uniform(-1, 1, size=dimension)
    x_opt = convex_set.project(x_0)

    A = scipy.sparse.rand(m=shape[0], n=shape[1], density=density, format='csc')
    b = A.dot(x_opt)
        
    return FeasibilityProblem([convex_set, AffineSet(convex_set._x, A, b)],
        x_opt)
