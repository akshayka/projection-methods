import cvxpy as cvxpy
import numpy as np
import unittest

class TestProjectionStability(unittest.TestCase):
    def test_projection(self):
        """Test whether CVXPY works as expected wrt NumPy shapes."""
        x = cvxpy.Variable(1000)
        x_0 = np.ones(1000)
        soc_constr = [cvxpy.norm(x[:-1], 2) <= x[-1]]

        obj = cvxpy.Minimize(cvxpy.norm(x_0 - x, 2))
        prob = cvxpy.Problem(obj, soc_constr)
        prob.solve(solver=cvxpy.ECOS)
        self.assertTrue(x.value.shape == (1000, 1))
        x_star = np.array(x.value).flatten()
        self.assertTrue(x_star.shape == (1000,), x_star.shape)
        self.assertEqual(np.linalg.norm(x_0 - x_star, 2), obj.value)
        self.assertTrue(np.isclose(prob.value, obj.value, atol=1e-7))
