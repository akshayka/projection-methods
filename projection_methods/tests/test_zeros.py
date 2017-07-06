import cvxpy as cvxpy
import numpy as np
import unittest

from projection_methods.oracles.zeros import Zeros
import projection_methods.tests.utils as utils

class TestZeros(unittest.TestCase):
    def test_projection(self):
        """Test projection, query methods of the Zeros oracle."""
        x = cvxpy.Variable(1000)
        zeros = Zeros(x)
        constr = zeros._constr

        # Test idempotency
        x_0 = np.zeros(1000)
        x_star = zeros.project(x_0)
        self.assertTrue(zeros.contains(x_star))
        self.assertTrue(np.array_equal(x_0, x_star), "projection not "
            "idemptotent")
        p = cvxpy.Problem(cvxpy.Minimize(cvxpy.pnorm(x - x_0, 2)), constr)
        p.solve()
        self.assertTrue(np.isclose(np.array(x.value).flatten(), x_star,
            atol=1e-3).all())
        utils.query_helper(self, x_0, x_star, zeros, idempotent=True)
        

        # Test the case in which x_0 >= 0
        x_0 = np.abs(np.random.randn(1000))
        x_star = zeros.project(x_0)
        self.assertTrue(np.array_equal(x_star, np.zeros(1000)),
            "projection not  zero")
        p = cvxpy.Problem(cvxpy.Minimize(cvxpy.pnorm(x - x_0, 2)), constr)
        p.solve()
        self.assertTrue(np.isclose(np.array(x.value).flatten(), x_star,
            atol=1e-3).all())
        utils.query_helper(self, x_0, x_star, zeros, idempotent=False)

        # Test the case in which x_0 <= 0
        x_0 = -1 * np.abs(np.random.randn(1000))
        x_star = zeros.project(x_0)
        self.assertTrue(np.array_equal(x_star, np.zeros(1000)),
            "projection not  zero")
        p = cvxpy.Problem(cvxpy.Minimize(cvxpy.pnorm(x - x_0, 2)), constr)
        p.solve()
        self.assertTrue(np.isclose(np.array(x.value).flatten(), x_star,
            atol=1e-3).all())
        utils.query_helper(self, x_0, x_star, zeros, idempotent=False)

        # Test random projection
        x_0 = np.random.randn(1000)
        x_star = zeros.project(x_0)
        self.assertTrue(np.array_equal(x_star, np.zeros(1000)),
            "projection not  zero")
        p = cvxpy.Problem(cvxpy.Minimize(cvxpy.pnorm(x - x_0, 2)), constr)
        p.solve()
        self.assertTrue(np.isclose(np.array(x.value).flatten(), x_star,
            atol=1e-3).all())
