import argparse
import cPickle
import cvxpy
from pathlib2 import PosixPath
import sys

from projection_methods.oracles.nonneg import NonNeg
from projection_methods.oracles.soc import SOC
from projection_methods.oracles.zeros import Zeros
from projection_methods.problems.problem_factory import random_cone_program


# TODO(akshayka): refactor and remove from here / save_convex_affine_problem
k_zeros = 'Z'
k_soc = 'SOC'
k_nn = 'NN'
k_cones = {k_zeros: Zeros, k_soc: SOC, k_nn: NonNeg}

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
        '-cd', '--cone_dims', required=True, type=int, nargs='+',
        help='list of cone dimensions')
    parser.add_argument(
        '-c', '--cones', required=True, type=str, nargs='+',
        help='list of cone classes belonging to %s' % str(k_cones.keys()))
    parser.add_argument(
        '-n', '--n', required=True, type=int,
        help='number of variables (i.e., dimension of p)')
    parser.add_argument(
        '-d', '--density', type=float, default=.01,
        help='density of data matrix A')
    args = parser.parse_args()

    path = PosixPath(args.output).expanduser()
    # TODO(akshayka): refactor and remove
    die_if(path.is_dir(), 'Please enter a filename, not a directory.')
    die_if(not path.parents[0].is_dir(), 'You are trying to save your '
        'problem in a non-extant directory.')
    die_if(path.is_file(), 'You are trying to overwrite an extant file; '
        'this is not allowed.')

    dim = 2 * (sum(args.cone_dims) + args.n + 1)
    x = cvxpy.Variable(dim)

    for c in args.cones:
        assert c in k_cones
    cones = [k_cones[c] for c in args.cones]
    cone_program = random_cone_program(x=x, cone_dims=args.cone_dims,
        cones=cones, n=args.n, density=args.density)

    with path.open('wb') as f:
        cPickle.dump(cone_program, f, protocol=cPickle.HIGHEST_PROTOCOL)

    with open(str(path) + '.txt', 'wb') as f:
        f.write('-cone_dims: %s\n' % str(args.cone_dims))
        f.write('-cones: %s\n' % args.cones)
        f.write(str(cone_program))

    print 'Saved problem at ' + str(path)
        

if __name__ == '__main__':
    main()
