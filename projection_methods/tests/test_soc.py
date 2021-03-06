import cvxpy as cvxpy
import numpy as np
import unittest

from projection_methods.oracles.soc import SOC
import projection_methods.tests.utils as utils

class TestSOC(unittest.TestCase):
    def test_projection(self):
        """Test projection, query methods of the SOC oracle."""
        x = cvxpy.Variable(1000)
        x_0 = np.ones(1000)
        soc = SOC(x)
        constr = soc._constr

        # Test idempotency
        z = x_0[:-1]
        norm_z = np.linalg.norm(x_0[:-1], 2)
        x_0[-1] = norm_z + 10
        x_star = soc.project(x_0)
        self.assertTrue(soc.contains(x_star))
        self.assertTrue(np.array_equal(x_0, x_star), "projection not "
            "idemptotent")
        p = cvxpy.Problem(cvxpy.Minimize(cvxpy.pnorm(x - x_0, 2)), constr)
        p.solve()
        self.assertTrue(np.isclose(np.array(x.value).flatten(), x_star,
            atol=1e-3).all())
        utils.query_helper(self, x_0, x_star, soc, idempotent=True)
        

        # Test the case in which norm_z <= -t
        x_0[-1] = -1 * norm_z
        x_star = soc.project(x_0)
        self.assertTrue(np.array_equal(x_star, np.zeros(1000)),
            "projection not  zero")
        p = cvxpy.Problem(cvxpy.Minimize(cvxpy.pnorm(x - x_0, 2)), constr)
        p.solve()
        self.assertTrue(np.isclose(np.array(x.value).flatten(), x_star,
            atol=1e-3).all())
        utils.query_helper(self, x_0, x_star, soc, idempotent=False)

        # Test the case in which norm_z > abs(t)
        x_0[-1] = norm_z - 10
        x_star = soc.project(x_0)
        p = cvxpy.Problem(cvxpy.Minimize(cvxpy.pnorm(x - x_0, 2)), constr)
        p.solve()
        self.assertTrue(np.isclose(np.array(x.value).flatten(), x_star,
            atol=1e-3).all())
        utils.query_helper(self, x_0, x_star, soc, idempotent=False)

        # Test random projection
        x_0 = np.random.randn(1000)
        x_star = soc.project(x_0)
        p = cvxpy.Problem(cvxpy.Minimize(cvxpy.pnorm(x - x_0, 2)), constr)
        p.solve()
        self.assertTrue(np.isclose(np.array(x.value).flatten(), x_star,
            atol=1e-3).all())
