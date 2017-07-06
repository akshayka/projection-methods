import cvxpy as cvxpy
import numpy as np
import unittest

from projection_methods.oracles.affine_set import AffineSet
import projection_methods.tests.utils as utils

class TestAffineSet(unittest.TestCase):
    def test_projection(self):
        """Test projection, query methods of the AffineSet oracle."""
        m = 150
        n = 200
        x = cvxpy.Variable(n)

        sol = np.random.randn(n)
        A = np.random.randn(m, n)
        b = A.dot(sol)

        affine = AffineSet(x, A, b)
        constr = affine._constr

        # Test idempotency
        x_0 = sol
        x_star = affine.project(x_0)
        self.assertTrue(affine.contains(x_star))
        self.assertTrue(np.array_equal(x_0, x_star), "projection not "
            "idemptotent")
        p = cvxpy.Problem(cvxpy.Minimize(cvxpy.pnorm(x - x_0, 2)), constr)
        p.solve()
        self.assertTrue(np.isclose(np.array(x.value).flatten(), x_star,
            atol=1e-3).all())
        utils.query_helper(self, x_0, x_star, affine, idempotent=True)
        
        # Test random x_0
        x_0 = np.random.randn(n)
        while (np.isclose(A.dot(x_0), np.zeros(m)).all()):
            x_0 = np.random.randn(n)
        x_star = affine.project(x_0)
        self.assertTrue(np.isclose(A.dot(x_star), b).all())
        p = cvxpy.Problem(cvxpy.Minimize(cvxpy.pnorm(x - x_0, 2)), constr)
        p.solve()
        self.assertTrue(np.isclose(np.array(x.value).flatten(), x_star,
            atol=1e-3).all())
        utils.query_helper(self, x_0, x_star, affine, idempotent=False)
