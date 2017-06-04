from projection_methods.oracles.convex_set import ConvexSet


class FeasibilityProblem(object):
    """Description of a convex feasibility problem

    Defines a convex feasibility problem, i.e.,
        find x
        s.t. x \in C_1 \cap C_2

    Attributes:
        sets (list of ConvexSet): the two convex sets defining the
            feasibility problem
        x_opt (array-like): any point in the intersection of the sets
        unique (boolean): whether the sets share exactly one point
        dimension (tuple): dimension of the space in which x lies
    """
    def __init__(self, sets, x_opt, unique=True):
        """
        Args:
            sets (list of ConvexSet): the convex sets defining the
            feasibility problem
            x_opt (array-like): any point in the intersection of the sets

        Keyword Args:
            unique (boolean): whether the sets share exactly one point
                (default True)
        """

        if len(sets) != 2:
            raise ValueError('Feasibility problems must be cast as finding a '
                'point in the intersection of _exactly_ two convex sets.')
        assert isinstance(sets[0], ConvexSet)
        assert isinstance(sets[1], ConvexSet)
        self.sets = sets
        self.x_opt = x_opt
        self.dimension = x_opt.shape
        self.unique = unique
