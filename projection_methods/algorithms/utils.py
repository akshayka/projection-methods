import cvxpy as cvx
import logging
import numpy as np

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
    -------
    The projection of point onto cvx_set : list-like (float)
    """
    value, _ = project_aux(point=point, cvx_set=cvx_set, cvx_var=cvx_var)
    return value


def project_aux(point, cvx_set, cvx_var, solver=cvx.SCS, use_indirect=True):
    """Return the projection and the distance"""
    # If the projection is unconstrained, the projection is trivially
    # idempotent
    if cvx_set == []:
        return point, 0

    obj = cvx.Minimize(cvx.pnorm(cvx_var - point, 2))
    prob = cvx.Problem(obj, cvx_set)

    # ECOS sometimes fails at high tolerances
    #
    # SCS (indirect == True) sometimes returns values that are
    #  1. an order of magnitude different from the explicitly computed distance,
    #  2. and practically every value is somewhat different from the latter
    # SCS (indirect == False) has problem 2. of (SCS indirect == True)
    if solver == cvx.SCS:
        opt_dist = prob.solve(solver=cvx.SCS, use_indirect=use_indirect)
    else:
        try:
            tol = 1e-3
            opt_dist = prob.solve(
                solver=cvx.ECOS, abstol=tol, reltol=tol, feastol=tol)
        except cvx.error.SolverError:
            tol *= 10
            logging.warning(
                'ECOS failed with tol %f; retrying with tol %f', tol / 10, tol)
            opt_dist = prob.solve(
                solver=cvx.ECOS, abstol=tol, reltol=tol, feastol=tol)

    # numpy computes distances that are more accurate than those computed by
    # cvxp
    np_dist = np.linalg.norm(cvx_var.value - point, 2)
    if not np.isclose(np_dist, opt_dist, atol=1e-2):
        logging.warning('opt_dist (%f) != np_dist (%f)', opt_dist, np_dist)
    else:
        logging.debug('opt_dist and np_dist agree')

    if prob.status != cvx.OPTIMAL and prob.status != cvx.OPTIMAL_INACCURATE:
        logging.warning('problem status %s', prob.status)

    # distances computed explicitly with numpy are more reliable than those
    # computed by cvxpy, so return the former
    return cvx_var.value, np_dist


def plane_search(iterates, num_iterates, cvx_set, cvx_var):
    """
    Plane search on previous iterates when performing the projection.

    This plane search does not preserve Fejer monotonicity!

    Parameters
    ----------
    iterates : list-like (list-like (float))
        List of the iterates belonging to a convex set, where iterates[i] is
        the i-th iterate.
    num_iterates : int
        Use the num_iterates most recent iterates in the plane search
    cvx_set : list
        List of cvxpy constraints defining the convex set on which to project
    cvx_var : cvxpy.Variable
        cvxpy variable used in specifying cvx_set
    """
    num_iterates = min(len(iterates), num_iterates)
    iterates = iterates[-num_iterates:]
    theta = cvx.Variable(num_iterates)

    # in the plane search, we seek a convex combination of the iterates
    # that is close to a point in the other convex set
    obj = cvx.Minimize(
        cvx.pnorm(
            sum([p * it for (p, it) in zip(theta, iterates)]) - cvx_var, 2))
    constrs = cvx_set + [cvx.sum_entries(theta) == 1] + [theta >= 0] + [theta <= 1]
    prob = cvx.Problem(obj, constrs)
    prob.solve(solver=cvx.SCS, use_indirect=True)
    opt_point = sum([p.value * it for (p, it) in zip(theta, iterates)])
    np_dist = np.linalg.norm(opt_point - cvx_var.value, 2)
    return opt_point, np_dist

def momentum_update(iterates, velocity, alpha=0.8, beta=0.2):
    """
    Move with momentum to the next iterate subject to the velocity

    Parameters
    ----------
    iterates : list-like (list-like (float))
       List of all the iterates so far, where iterates[i] is the i-th iterate.
    velocity : list-like (float)
        The velocity with which the current iterate should move:
            next_iterate - iterate, or gradient at iterates[-1]
        for example. An (imperfect) phyiscal interpretation is that, without
        momentum the iterate's new position would be
            iterates[-1] + time_step * velocity.
    Returns
    -------
    The next iterate : list-like (float)
    """
    if len(iterates) < 2:
        return iterates[-1] + velocity
    else:
        momentum = iterates[-1] - iterates[-2]        
        return iterates[-1] + alpha * velocity + beta * momentum
