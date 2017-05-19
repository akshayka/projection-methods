import abc

import cvxpy
import numpy as np

import algorithms.utils


class Projectable(object):
    def __init__(self, x, constr=[]):
        self._x = x
        self._constr = constr
        variables = set([v for c in constr for v in c.variables()])
        assert len(variables) <= 1, ("ConvexSet expects at most one variable "
            "to be constrained among its constraints, "
            "received %d" % len(variables))
        if len(variables) == 1:
            constrained = variables.pop()
            assert constrained is self._x, ("Constrained "
                "variable must be exactly the variable supplied to __init__; "
                "constrained name: %s, supplied name: %s" %
                (constrained._name, self._x._name))


    def project(self, x_0):
        x_star = algorithms.utils.project(x_0, self._constr, self._x)
        return x_star


    def contains(self, x_0):
        return all(np.isclose(self.project(x_0), x_0, atol=1e-3))


    def __repr__(self):
        string = type(self).__name__ + "\n"
        for c in self._constr:
            string += c.__str__() + "; "
        return string


class Halfspace(Projectable):
    def __init__(self, x, a, b):
        self._a = a
        self._b = b
        constr = [a.T * x <= b]
        super(Halfspace, self).__init__(x, constr)


    def __repr__(self):
        string = type(self).__name__ + "\n"
        string += "a: " + str(self._a) + "\n"
        string += "b: " + str(self._b)
        return string
        

class Hyperplane(Projectable):
    def __init__(self, x, a, b):
        self._a = a
        self._b = b
        constr = [a.T * x == b]
        super(Hyperplane, self).__init__(x, constr)


    def __repr__(self):
        string = type(self).__name__ + "\n"
        string += "a: " + str(self._a) + "\n"
        string += "b: " + str(self._b)
        return string


class Polyhedron(Projectable):
    class Eviction(object):
        LRA, RANDOM, MRA = range(3)


    def __init__(self, x, information=[],
            max_hyperplanes=float("inf"),
            max_halfspaces=float("inf"),
            eviction_policy=Eviction.LRA):
        self._hyperplanes = []
        self._halfspaces = []
        self.max_hyperplanes = max_hyperplanes
        self.max_halfspaces = max_halfspaces
        self.add(information)
        self._eviction_policy = eviction_policy
        constr = [c for h in self._hyperplanes for c in h._constr]
        constr += [c for h in self._halfspaces for c in h._constr]
        super(Polyhedron, self).__init__(x, constr)


    def add(self, items):
        if type(items) is not list:
            items = [items]
        for item in items:
            if type(item) == Hyperplane:
                self._add_hyperplane(item)
            elif type(item) == Halfspace:
                self._add_halfspace(item)
            else:
                raise ValueError, "Only Halfspaces or Hyperplanes can be added"


    def halfspaces(self):
        return self._halfspaces


    def hyperplanes(self):
        return self._hyperplanes


    def _add_hyperplane(self, hyperplane):
        if len(self._hyperplanes) == self.max_hyperplanes:
            # TODO(akshayka): implement eviction
            pass
        self._hyperplanes.append(hyperplane)


    def _add_halfspace(self, halfspace):
        if len(self._halfspaces) == self.max_halfspaces:
            # TODO(akshayka): implement eviction
            pass
        self._halfspaces.append(halfspace)



class Oracle:
    __metaclass__ = abc.ABCMeta
    class Outer(object):
        POLYHEDRAL, EXACT = range(2)

    
    @abc.abstractmethod
    def query(self, x_0):
        pass


    @abc.abstractmethod
    def outer(self, kind=Outer.POLYHEDRAL):
        pass

        
class ConvexSet(Projectable, Oracle):
    def __init__(self, x, constr=[]):
        self._halfspaces = []
        super(ConvexSet, self).__init__(x, constr)


    def query(self, x_0):
        if self.contains(x_0):
            return x_0, []
        x_star = self.project(x_0)
        a = x_0 - x_star
        b = a.dot(x_star)
        halfspace = Halfspace(x=self._x , a=a, b=b) 
        self._halfspaces.append(halfspace)
        return x_star, halfspace


    def outer(self, kind=Oracle.Outer.POLYHEDRAL):
        if kind == Oracle.Outer.POLYHEDRAL:
            return Polyhedron(self._x, self._halfspaces)
        elif kind == Oracle.Outer.EXACT:
            return self
        else:
            raise(ValueError, "Unknown kind " + str(kind))


class AffineSet(ConvexSet):
    def __init__(self, x, A, b):
        constr = [A * x == b]
        self._A = A
        self._b = b
        self._hyperplanes = []
        super(AffineSet, self).__init__(x, constr)


    def query(self, x_0):
        x_star = self.project(x_0)
        a = x_0 - x_star
        b = a.dot(x_star)
        hyperplane = Hyperplane(x=self._x , a=a, b=b) 
        self._hyperplanes.append(hyperplane)
        return x_star, hyperplane

    def __repr__(self):
        string = type(self).__name__ + "\n"
        string += "A: " + str(self._A) + "\n"
        string += "b: " + str(self._b)
        return string


class SOC(ConvexSet):
    def __init__(self, x):
        constr = [cvxpy.norm(x[:-1], 2) <= x[-1]]
        super(SOC, self).__init__(x, constr)


class CartesianProduct(ConvexSet):
    def __init__(sets, dims):
        """
            dims: a list of slice objects
        """
        self._sets = sets
        x = self._sets[0]._x
        constr = [c for s in sets for c in s._constr]
        super(CartesianProduct, self).__init__(x, constr)


    def query(x_0):
        x_star = np.zeros(x_0.shape)
        halfspaces = []
        for s, dim in zip(self._sets, self._dims):
            x_s, h_s = s.query(x_0[dim])
            x_star[dim] = x_s
            halfspaces.append(h_s)
        self._halfspaces.extend(halfspaces)
        return x_star, halfspaces
