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
        self.cvx_sets = cvx_sets
        self.cvx_var = cvx_var
        self.var_dim = var_dim
