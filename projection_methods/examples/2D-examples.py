from projection_methods.algorithms.alternating_projections import AlternatingProjections
from projection_methods.algorithms.qp_solver import QPSolver
from projection_methods.problems import FeasibilityProblem

import argparse
import cvxpy as cvx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-l', help='lines', action='store_true')
parser.add_argument('-c', help='circles', action='store_true')
parser.add_argument(
    '-ps', '--plane_search', type=int, default=1,
    help='# iterates to include when performing a plane search on the '\
         'affine set; 1 ==> no plane search')
args = vars(parser.parse_args())


def plot(iterates, m=None):
    x = [float(i[0]) for i in iterates]
    y = [float(i[1]) for i in iterates]
    plt.scatter(x=x, y=y)
    plt.plot(x, y)

    if args['l']:
        x_val = (100 / m) * 2
        plt.scatter([-1 * x_val, 0, x_val], [x_val * m, 0, x_val * m])
        plt.plot([-1 * x_val, 0, x_val], [x_val * m, 0, x_val * m])

# Intersection of y=m and y=-mx
if args['l']:
    m = 20
    x = cvx.Variable(2)
    a = np.array([[m], [-1]])
    b = np.array([[m], [1]])
    point = np.array([[0], [100]])
    cvx_sets = [[a.T * x == 0], [b.T * x == 0]]
    problem = FeasibilityProblem(cvx_sets=cvx_sets, cvx_var=x, var_dim=2)
    initial_point = np.array([[0], [100]])
    max_iters = 100

    plt.figure()
    solver = AlternatingProjections(
        initial_point=initial_point, max_iters=max_iters,
        plane_search=[args['plane_search'], 1])
    solver.solve(problem)
    plot(solver.iterates, m)
    plt.title('Alternating projections')

    plt.figure()
    plt.plot([np.linalg.norm(x) for x in np.ediff1d(solver.iterates)])
    plt.title('Alternating projections: sequence of normed residuals')

    plt.figure()
    solver = QPSolver(initial_point=initial_point, max_iters=max_iters)
    solver.solve(problem)
    plot(solver.iterates, m)
    plt.show()

if args['c']:
    # ------ Circles ------
    x = cvx.Variable(2)
    r = 10
    options = {}
    options['initial_point'] = np.array([[0], [r]])
    options['max_iters'] = 10
    cvx_sets = [[cvx.square(x[0] - r) + cvx.square(x[1]) <= r**2],
                [cvx.square(x[0] + r) + cvx.square(x[1]) <= r**2]]
    problem = FeasibilityProblem(cvx_sets=cvx_sets, cvx_var=x, var_dim=2)

    # plot circles
    def plot_circles(fig):
        ax = fig.add_subplot(111, aspect='equal')
        ax.add_patch(
            patches.Circle(
                (-r, 0),
                r,
                fill=False      # remove background
            )
        )
        ax.add_patch(
            patches.Circle(
                (r, 0),
                r,
                fill=False      # remove background
            )
        )

    plot_circles(plt.figure())
    solver = AlternatingProjections()
    solver.solve(problem, options)
    plot(solver.iterates)

    plot_circles(plt.figure())
    solver = QPSolver()
    solver.solve(problem, options)
    plot(solver.iterates)
    plt.show()
