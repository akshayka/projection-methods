import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from projection_methods.oracles.convex_set import ConvexSet
from projection_methods.projectables.hyperplane import Hyperplane

class AffineSet(ConvexSet):
    """An oracle for affine sets
    
    Defines an oracle for affine sets of the form
        \{ x | Ax = b \},
    parametrized by A, b

    Attributes:
        x (cvxpy.Variable): a symbolic representation of
            members of the set
        A (numpy.ndarray or scipy.sparse matrix): a matrix
        b (numpy.ndarray): a target vector
    """
    def __init__(self, x, A, b):
        """
        Args:
            x: see 
            A (numpy.ndarray): a matrix
            b (numpy.ndarray): a target vector
        """
        assert A.shape[1] == x.size[0]
        constr = [A * x == b]
        self.A = A
        self.b = b
        super(AffineSet, self).__init__(x, constr)
        self._kkt_solver = None
        self.chosen_rows = set([])


    def contains(self, x_0, atol=1e-4):
        """Return True if x_0 in affine set, False otherwise"""
        return np.allclose(self.A.dot(x_0), self.b, atol=atol)


    def _make_kkt_solver(self):
        sparse_eye = scipy.sparse.eye(self.A.shape[1], format='csc')
        kkt_matrix = scipy.sparse.bmat(
            [[sparse_eye, self.A.T], [self.A, None]], format='csc')
        kkt_solver = scipy.sparse.linalg.factorized(kkt_matrix)
        return kkt_solver


    def project(self, x_0):
        if self.contains(x_0):
            return x_0

        if self._kkt_solver is None:
            # TODO(akshayka): it would be fine to do this in init,
            # except for whatever reason the return value of factorized()
            # cannot be pickled
            # ("expected string or Unicode object, NoneType found")
            self._kkt_solver = self._make_kkt_solver()
        target = np.hstack((x_0, self.b))
        sol = self._kkt_solver(target)
        return sol[:self._shape[0]]
        

    def query(self, x_0, data_hyperplanes=0, policy='random'):
        """As ConvexSet.query, but returns a Hyperplane

        Args:
            x_0 (array-like): query point
            data_hyperplanes: number of data hyperplanes to include per query
                              call
            policy: policy to use when gathering data hyperplanes; one of
                \{'random', 'largest_residual'\}.
        Returns:
            array-like: the projection of x_0 onto the set
            list of Hyperplane: a hyperplane of the form <a, x> = b
                in which every point x in the affine set must lie
        """
        x_star = self.project(x_0)
        if np.array_equal(x_star, x_0):
            return x_0, []

        hyperplanes = []
        # a.dot(y - x_star) == 0, for all y in affine set
        # <==> a.dot(y) == a.dot(x_star)
        a = x_0 - x_star
        b = a.dot(x_star)
        if abs(b) < 1e-7:
            # If the affine set is in fact a subspace, this
            # will always be triggered.
            b = np.array([0])
        hyperplanes.append(Hyperplane(x=self._x , a=a, b=b))

        if data_hyperplanes > 0:
            if policy == 'random':
                indices = np.random.permutation(np.prod(self.b.shape))
            elif policy == 'largest_residual':
                # sort the indices of the residuals in decreasing order
                r = np.abs(self.A.dot(x_0) - self.b)
                indices = np.flipud(np.argsort(r))
            else:
                raise ValueError('Unknown policy %s' % policy)
            num_chosen = 0
            for idx in indices:
                if num_chosen >= data_hyperplanes:
                    break
                if idx not in self.chosen_rows:
                    self.chosen_rows.add(idx)
                    num_chosen += 1
                    hyperplanes.append(Hyperplane(x=self._x,
                        a=self.A.getrow(idx).T, b=np.array(self.b[idx])))
        self._info.extend(hyperplanes)
        return x_star, hyperplanes

    def __repr__(self):
        string = type(self).__name__ + '\n'
        string += 'A of shape %s\n' % str(self.A.shape)
        string += 'b of shape %s' % str(self.b.shape)
        return string
