"""
Problem types.
"""


class FeasibilityProblem(object):
    """Find point in affine_set \cap cone

    attributes
    ----------
    """
    def __init__(self, cvx_sets, cvx_var, var_dim):
        # TODO(akshayka): I will need to refactor everything to eliminate cvx_var
        if len(cvx_sets) != 2:
            raise ValueError('Feasibility problems must be cast as finding a '
                'point in the intersection of _exactly_ two convex sets.')
        self.cvx_sets = cvx_sets
        self.cvx_var = cvx_var
        self.var_dim = var_dim
