from projection_methods.algorithms.alternating_projections import AlternatingProjections
from projection_methods.algorithms.qp_solver import QPSolver
from projection_methods.problems import FeasibilityProblem

import cvxpy as cvx
import matplotlib.pyplot as plt
import numpy as np

# Intersection of y=m and y=-mx
m = 20
x = cvx.Variable(2)
a = np.array([[m], [-1]])
b = np.array([[m], [1]])
point = np.array([[0], [100]])
cvx_sets = [a.T * x == 0, b.T * x == 0]
problem = FeasibilityProblem(cvx_sets=cvx_sets, cvx_var=x, var_dim=2)
options = {'initial_point': np.array([[0], [100]]), 'max_iters': 100}

def plot(iterates, m):
    x = [float(i[0]) for i in iterates]
    y = [float(i[1]) for i in iterates]
    plt.figure()
    plt.scatter(x=x, y=y)
    plt.plot(x, y)

    x_val = (100 / m) * 2
    plt.scatter([-1 * x_val, 0, x_val], [x_val * m, 0, x_val * m])
    plt.plot([-1 * x_val, 0, x_val], [x_val * m, 0, x_val * m])

solver = AlternatingProjections()
iterates = solver.solve(problem, options)
plot(iterates, m)

solver = QPSolver()
iterates = solver.solve(problem, options)
plot(iterates, m)
plt.show()
