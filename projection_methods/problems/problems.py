from projection_methods.oracles.affine_set import AffineSet
from projection_methods.oracles.cartesian_product import CartesianProduct
from projection_methods.oracles.convex_set import ConvexSet
from projection_methods.oracles.nonneg import NonNeg
from projection_methods.oracles.reals import Reals
from projection_methods.oracles.cone import Cone
from projection_methods.oracles.zeros import Zeros


class FeasibilityProblem(object):
    """Description of a convex feasibility problem

    Defines a convex feasibility problem, i.e.,
        find x
        s.t. x \in C_1 \cap C_2

    Attributes:
        sets (list of ConvexSet): the two convex sets defining the
            feasibility problem
        x_opt (array-like): any point in the intersection of the sets
        dimension (tuple): dimension of the space in which x lies
    """
    def __init__(self, sets, x_opt):
        """
        Args:
            sets (list of ConvexSet): the convex sets defining the
            feasibility problem
            x_opt (array-like): any point in the intersection of the sets
        """

        if len(sets) != 2:
            raise ValueError('Feasibility problems must be cast as finding a '
                'point in the intersection of _exactly_ two convex sets.')
        assert isinstance(sets[0], ConvexSet)
        assert isinstance(sets[1], ConvexSet)
        self.sets = sets
        self.x_opt = x_opt
        self.dimension = x_opt.shape


    def __repr__(self):
        string = type(self).__name__ + '\n'
        string += 'dimension: %d\n' % self.dimension
        string += str(self.sets[0]) + '\n'
        string += str(self.sets[1])
        return string


class SCSProblem(FeasibilityProblem):
    """Description of a homogeneous self-dual embedding for cone programs

    Defines a problem of the form targeted by the Splitting Conic Solver
    [see http://web.stanford.edu/~boyd/papers/scs.html]. In particular,
    it forms a feasibility problem of the form
    (1) find (u, v)
        s.t. Qu = v,
             (u, v) \in C \times C^*
    where 
        Q = [[0, A.T]; [A, 0]],
        u = (p, y, tau),
        v = (r, s, kappa),
        C   = R^n \times K^* \times R_+, and
        C^* = {0}^n \times K \times R_+, K an m-dimensional cone.

    As explained in the linked-to reference, this feasibility problem
    corresponds to the cone program (with variable p)
    (2) min. c.T * p
        s.t. Ap + s = b
             s \in K.

    Note that the iterate of problem (1) (and thus of this problem) is a
    variable uv = (u, v). Also note that this problem _always_ admits zero as a
    solution, so in general this problem does not have a unique solution.

    Attributes:
        sets (list of ConvexSet): the two convex sets defining the feasibility
            problem, sets[0] the affine set [Q, -I] * [u, v].T = 0, set[1]
            the cross product of the cones.
        x_opt (array-like): any point (u, v) that is optimal for (1)
        Q (np.array): the augmented KKT matrix, as defined above
        A (np.array): as defined above
        b (np.array): as defined above
        c (np.array): as defined above
        p_opt (array-like): any point that is optimal for (2)
    """
    def __init__(self, sets, x_opt, Q, A, b, c, p_opt=None):
        """
        Args:
            sets (list of ConvexSet): the convex sets defining the
            feasibility problem
            x_opt (array-like): any point in the intersection of the sets
                (i.e., any point that is optimal for (1))
            Q (np.array): as defined abnove
            A (np.array): as defined above 
            b (np.array): as defined above
            c (np.array): as defined above
            p_opt (array-like): any point that is optimal for (2) (optional)
        TODO(akshayka): it is unwieldly to require the user to pass in
        the constructed sets, as well as the problem data Q. But requiring
        them to do so allows us to isolate this class from how sets are
        defined.
        """

        # problem data corresoponding to (1)
        if len(sets) != 2:
            raise ValueError('Feasibility problems must be cast as finding a '
                'point in the intersection of _exactly_ two convex sets.')

        # Ensure that the cartesian product is in the correct form
        assert isinstance(sets[0], CartesianProduct)
        self.product_set = sets[0]
        assert len(self.product_set.sets) == 6
        assert isinstance(self.product_set.sets[0], Reals)
        assert isinstance(self.product_set.sets[1], Cone)
        assert isinstance(self.product_set.sets[2], NonNeg)
        assert isinstance(self.product_set.sets[3], Zeros)
        assert isinstance(self.product_set.sets[4], Cone)
        assert isinstance(self.product_set.sets[5], NonNeg)

        assert isinstance(sets[1], AffineSet)

        self.sets = sets
        self.x_opt = x_opt
        self.dimension = sets[0]._var.size[0]
        self.n = self.product_set.sets[0]._x.size[0]
        self.m = self.product_set.sets[1]._x.size[0]

        # problem data corresponding to (2); i.e., the provenance of (1)
        self.Q = Q
        self.A = A
        self.c = c
        self.b = b
        self.p_opt = p_opt

    # utility functions for computing the objective function of problem (2)
    # and, if p_opt was provided, for retrieving the optimal value of (2)
    def objective_value(self, p):
        return self.c.dot(p) 

    def optimal_value(self):
        return self.c.dot(self.p_opt) if self.p_opt is not None else None

    # utility functions for extracting the individual components of (u, v);
    #
    # recall that u := (p, y, tau)
    def p(self, uv):
        assert uv.shape == (self.dimension,)
        return uv[self.product_set.slices[0]]

    def y(self, uv):
        assert uv.shape == (self.dimension,)
        return uv[self.product_set.slices[1]]

    def tau(self, uv):
        assert uv.shape == (self.dimension,)
        return float(uv[self.product_set.slices[2]])

    # v := (r, s, kappa)
    def r(self, uv):
        assert uv.shape == (self.dimension,)
        return uv[self.product_set.slices[3]]

    def s(self, uv):
        assert uv.shape == (self.dimension,)
        return uv[self.product_set.slices[4]]

    def kappa(self, uv):
        assert uv.shape == (self.dimension,)
        return float(uv[self.product_set.slices[5]])


    def __repr__(self):
        string = type(self).__name__ + '\n'
        string += 'dimension: %d\n' % self.dimension
        string += 'm (cone shape / # constraints): %d\n' % self.m
        string += 'n (p shape / # variables): %d\n' % self.n
        string += str(self.sets[0]) + '\n'
        string += str(self.sets[1]) 
        return string
