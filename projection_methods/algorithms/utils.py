import cvxpy as cvx

def project(point, cvx_set, cvx_var):
    """ 
    Project onto a convex set.

    Parameters
    ----------
    point : double
        The point to project onto the provided set.
    cvx_set : list
        List of cvxpy constraints defining a convex set.
    cvx_var : cvxpy.Variable
        cvxpy variable used in specifying cvx_set

    Returns
    ------- The projection of point onto cvx_set.
    """
    obj = cvx.Minimize(cvx.norm(cvx_var - point, 2))
    prob = cvx.Problem(obj, cvx_set)
    prob.solve()
    return cvx_var.value
