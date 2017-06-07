import cvxpy
import logging
import numpy as np

def project(x_0, cvxpy_set, cvxpy_var):
    """ 
    Project onto a convex set.

    Parameters
    ----------
    x_0 : double
        The point to project onto the provided set.
    cvxpy_set : list
        List of cvxpy constraints defining a convex set.
    cvxpy_var : cvxpy.Variable
        cvxpy variable used in specifying cvxpy_set

    Returns
    -------
    The projection of x_0 onto cvxpy_set : list-like (float)
    """
    x_star, _ = project_aux(x_0=x_0, cvxpy_set=cvxpy_set, cvxpy_var=cvxpy_var)
    return x_star


def project_aux(x_0, cvxpy_set, cvxpy_var, solver=cvxpy.MOSEK,
    use_indirect=True, abstol=1e-4, reltol=1e-3, feastol=1e-4):
    """Return the projection and the distance"""
    # If the projection is unconstrained, the projection is trivially
    # idempotent
    if cvxpy_set == []:
        return x_0, 0

    obj = cvxpy.Minimize(cvxpy.pnorm(cvxpy_var - x_0, 2))
    prob = cvxpy.Problem(obj, cvxpy_set)

    # ECOS sometimes fails at high tolerances
    #
    # SCS (indirect == True) sometimes returns values that are
    #  1. an order of magnitude different from the explicitly computed distance,
    #  2. and practically every value is somewhat different from the latter
    # SCS (indirect == False) has problem 2. of (SCS indirect == True)
    if solver == cvxpy.MOSEK:
        solver_dist = prob.solve(solver=cvxpy.MOSEK)
    elif solver == cvxpy.SCS:
        solver_dist = prob.solve(solver=cvxpy.SCS, use_indirect=use_indirect)
    else:
        solved = False
        while abstol <= 1:
            try:
                solver_dist = prob.solve(solver=cvxpy.ECOS,
                    abstol=abstol, reltol=reltol, feastol=feastol)
                solved = True
                break
            except cvxpy.error.SolverError as e:
                abstol *= 10
                reltol *= 10
                feastol *= 10
                logging.warning(
                    'ECOS failed (%s) with tol %.1e; retrying with tol %.1e',
                    str(e), abstol / 10, abstol)
        if not solved:
            logging.warning('ECOS failed; falling back to SCS.')
            solver_dist = prob.solve(solver=cvxpy.SCS, use_indirect=use_indirect)

    x_star = np.array(cvxpy_var.value).flatten()
    np_dist = np.linalg.norm(x_star - x_0, 2)

    if not np.isclose(obj.value, np_dist):
        logging.warning('obj.value (%f) != np_dist (%f)', obj.value, np_dist)
    if not np.isclose(np_dist, solver_dist, atol=1e-5):
        logging.warning('solver_dist (%f) != np_dist (%f)',
            solver_dist, np_dist)

    if prob.status != cvxpy.OPTIMAL and prob.status != cvxpy.OPTIMAL_INACCURATE:
        logging.warning('problem status %s', prob.status)

    return x_star, np_dist


def plane_search(iterates, num_iterates, cvxpy_set, cvxpy_var):
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
    cvxpy_set : list
        List of cvxpy constraints defining the convex set on which to project
    cvxpy_var : cvxpy.Variable
        cvxpy variable used in specifying cvxpy_set
    """
    num_iterates = min(len(iterates), num_iterates)
    iterates = iterates[-num_iterates:]
    theta = cvxpy.Variable(num_iterates)

    # in the plane search, we seek a convex combination of the iterates
    # that is close to a point in the other convex set
    obj = cvxpy.Minimize(
        cvxpy.pnorm(
            sum([p * it for (p, it) in zip(theta, iterates)]) - cvxpy_var, 2))
    constrs = (cvxpy_set + [cvxpy.sum_entries(theta) == 1] +
        [theta >= 0] + [theta <= 1])
    prob = cvxpy.Problem(obj, constrs)
    prob.solve(solver=cvxpy.SCS, use_indirect=True)
    opt_point = sum([p.value * it for (p, it) in zip(theta, iterates)])
    np_dist = np.linalg.norm(opt_point - cvxpy_var.value, 2)
    return opt_point, np_dist

def heavy_ball_update(iterates, velocity, alpha=0.8, beta=0.2):
    """Heavy-ball momentum
    Move with momentum to the next iterate:

        x^{k+1} := x^k + \alpha * (P_{C}x^k - x^k) + \beta * (x^k - x^{k-1})

    where (P_{C}x^k - x^k) is the velocity and (x^k - x^{k-1}) the momentum

    Args:
        iterates (list-like of float): list of all the iterates so far, where
            iterates[i] is the i-th iterate.
        velocity (list-like of float): the velocity with which the current
            iterate should move:
                next_iterate - iterate, or gradient at iterates[-1]
            for example. An (imperfect) phyiscal interpretation is that,
            without momentum, the iterate's new position would be
                iterates[-1] + time_step * velocity, time_step == 1.
        alpha (float): Weight to assign the velocity towards the nominal
            iterate
        beta: float
            Weight to assign the momentum from the previous iterate to the
            current iterate

    Returns:
        The next iterate : list-like (float)
    """
    if len(iterates) < 2:
        return iterates[-1] + velocity
    else:
        momentum = iterates[-1] - iterates[-2]
        return iterates[-1] + alpha * velocity + beta * momentum
