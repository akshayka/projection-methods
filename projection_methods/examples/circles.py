import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from projection_methods.algorithms.altp import AltP
from projection_methods.algorithms.polyak import Polyak
from projection_methods.algorithms.apop import APOP
from projection_methods.oracles.convex_set import ConvexSet
from projection_methods.problems.problems import FeasibilityProblem


def plot_circles(r):
    fig = plt.gcf()
    ax = fig.gca()
    ax.set_xlim((-5, 5))
    ax.set_ylim((-1, 12))

    l_circle = plt.Circle((-r, 0), r, color='black', fill=False)
    r_circle = plt.Circle((r, 0), r, color='black', fill=False)
    ax.add_artist(l_circle)
    ax.add_artist(r_circle)

def plot_iterates(iterates, label):
    x = [float(i[0]) for i in iterates]
    y = [float(i[1]) for i in iterates]
    plt.scatter(x=x, y=y)
    plt.plot(x, y, label=label)


def main():
    x = cp.Variable(2)
    r = 10
    left_circle = ConvexSet(x, [cp.square(x[0] + r) + cp.square(x[1]) <= r**2])
    right_circle = ConvexSet(x, [cp.square(x[0] - r) + cp.square(x[1]) <= r**2])

    problem = FeasibilityProblem([left_circle, right_circle], np.array([0, 0]))

    initial_iterate = np.array([0, r])
    max_iters = 10

    plt.figure()
    plot_circles(r)
    solver = AltP(max_iters=max_iters, initial_iterate=initial_iterate)
    solver.solve(problem)
    plot_iterates(solver.all_iterates, 'Alternating Projections')

    solver = Polyak(max_iters=max_iters, initial_iterate=initial_iterate)
    it, res, status = solver.solve(problem)
    plot_iterates(it, 'Polyak\'s acceleration')

    solver = APOP(max_iters=max_iters, initial_iterate=initial_iterate,
        average=False)
    it, res, status = solver.solve(problem)
    plot_iterates(it, 'APOP')
    plt.title('Motivating APOP: Circles')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
