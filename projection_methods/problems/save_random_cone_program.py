import argparse
import cPickle
import cvxpy
from pathlib2 import PosixPath
import sys

from projection_methods.problems.problem_factory import random_cone_program


# TODO(akshayka): refactor to remove from this and save_convex_affine_problem.py
def die_if(cond, msg):
    if cond:
        print 'Error: ' + msg
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    # --- input/output --- #
    parser.add_argument(
        'output', metavar='O', type=str,
        help='output path specifying location in which to problem.')
    # --- problem parameters --- #
    parser.add_argument(
        'm', metavar='m', type=int,
        help='number of constraints m')
    parser.add_argument(
        'n', metavar='n', type=int,
        help='number of variables (i.e., dimension of p)')
    parser.add_argument(
        '-d', '--density', type=float, default=.01,
        help='density of data matrix A')
    args = parser.parse_args()
    args = parser.parse_args()

    path = PosixPath(args.output).expanduser()
    # TODO(akshayka): refactor out
    die_if(path.is_dir(), 'Please enter a filename, not a directory.')
    die_if(not path.parents[0].is_dir(), 'You are trying to save your '
        'problem in a non-extant directory.')
    die_if(path.is_file(), 'You are trying to overwrite an extant file; '
        'this is not allowed.')

    dim = 2 * (args.m + args.n + 1)
    x = cvxpy.Variable(dim)
    cone_program = random_cone_program(x=x, m=args.m, n=args.n)

    with path.open('wb') as f:
        cPickle.dump(cone_program, f, protocol=cPickle.HIGHEST_PROTOCOL)

    with open(str(path) + '.txt', 'wb') as f:
        f.write(str(cone_program))

    print 'Saved problem at ' + str(path)
        

if __name__ == '__main__':
    main()
