import numpy as np


def query_helper(test_case, x_0, x_star, convex_set, idempotent=False):
    x_prime, halfspaces = convex_set.query(x_0)
    test_case.assertTrue(np.array_equal(x_prime, x_star))
    test_case.assertTrue(convex_set.contains(x_prime))

    if not idempotent:
        test_case.assertTrue(len(halfspaces) == 1)
        x_halfspace = halfspaces[0].project(x_0)
        test_case.assertTrue(np.isclose(x_halfspace, x_prime, atol=1e-7).all())
        test_case.assertTrue(convex_set.contains(x_halfspace))
    else:
        test_case.assertTrue(len(halfspaces) == 0)

