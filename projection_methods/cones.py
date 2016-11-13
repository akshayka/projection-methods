from projection_methods.problems import FeasibilityProblem
from projection_methods.algorithms.utils import project

import cvxpy as cvx
import numpy as np
import scipy as sp

TYPES = frozenset(['reals', 'soc'])

class ConeTypes(object):
    REALS = 'reals'
    SOC = 'soc'
    TYPES = frozenset([REALS, SOC])


class Cone(object):
    """A representation of a convex cone

    A Cone instance represents the cross-product of some standard cones.

    Attributes
    ----------
    self.types : list
        A list of of cone types. Each type must be a member of ConeTypes.TYPES.
    self.dims : list
        A list of non-negative integers representing the dimensions of each
    """
    def __init__(self, types, dims, options={}):
        for cone_type in (t for t in types if t not in ConeTypes.TYPES):
            raise ValueError(
                'Invalid cone type (%s); type must be one of %s' % \
                 (cone_type, str(ConeTypes.TYPES)))

        self.types = types
        self.dims = dims
        slices = []
        prev = 0
        for dim in self.dims:
            slices.append(slice(prev, prev+dim))
            prev += dim
        self.slices = slices
        self.options = options


def cvx_constraints_for_cone(cone_type, cvx_var):
    if cone_type == ConeTypes.REALS:
        return []
    if cone_type == ConeTypes.SOC:
        return [cvx.norm(cvx_var[:-1], 2) <= cvx_var[-1]]
    elif cone_type in ConeType.TYPES:
        raise NotImplementedError('Cone %s not yet implemented.' % cone_type)
    else:
        raise ValueError('Cone %s not supported.' % cone_type)


def project_cone(point, cone):
    """Project point onto cone
    
    Convenience wrapper for project.
    """
    points = []
    for t, dim, slx in zip(cone.types, cone.dims, cone.slices):
        cvx_var = cvx.Variable(dim)
        points.append(project(point[slx],
            cvx_constraints_for_cone(t, cvx_var), cvx_var))
    return np.concatenate(points, axis=0)


def random_feasibility_problem(cone, affine_dim):
    """Generate a random feasibility problem

    The returned feasibility problem takes the form
        find x
        subject to Ax = b,
                   x in K
     where x \in R^{affine_dim, var_dim} and K is the cone,
     var_dim = sum [dim for dim in cone.dims]

    Parameters
    ----------
    cone : Cone
        The cone in which the variable should lie. Note that
        the cone is permitted to be the cross product of multiple
        convex cones. Any instance of Cone is a valid cone argument.
    affine_dim : int
        The number of rows that the data matrix A should have.

    Returns
    -------
    - A random, feasible instance of FeasibilityProblem.
    - a dict data with the problem data and an optimal solution
    """
    var_dim = sum(cone.dims)
    cvx_var = cvx.Variable(var_dim) 
    cvx_sets = []
    data = {}
    data['cone'] = cone

    # Generate a point x_k that lies in the provided cone.
    x_0 = np.random.uniform(-1, 1, size=(var_dim,1))
    x_opt = project_cone(x_0, cone)
    data['x_opt'] = x_opt

    # The affine constraint is of the form A * x_opt = b
    A = sp.sparse.rand(m=affine_dim, n=var_dim, density=0.01, format='csc')
    b = A * x_opt
    data['A'] = A
    data['b'] = b
    cvx_sets += [A * cvx_var == b]
 
    for t, slx in zip(cone.types, cone.slices):
        cvx_sets += cvx_constraints_for_cone(t, cvx_var[slx])

    return FeasibilityProblem(cvx_sets=cvx_sets, cvx_var=cvx_var,
        var_dim=var_dim), data
