"""
Problem types.
"""

import abc

class FeasibilityProblem(object):
    """Find point in affine_set \cap cone

    attributes:
    """
    def __init__(self, cvx_sets, cvx_var, var_dim):
        self.cvx_sets = cvx_sets
        self.cvx_var = cvx_var
        self.var_dim = var_dim
