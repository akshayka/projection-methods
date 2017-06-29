import argparse

from projection_methods.problems.problem_factory import random_linear_program
from projection_methods.problems.utils import check_path, save_problem


def main():
    parser = argparse.ArgumentParser()
    # --- input/output --- #
    parser.add_argument(
        'output', metavar='O', type=str,
        help='output path specifying location in which to problem.')
    # --- problem parameters --- #
    parser.add_argument(
        'm', metavar='rows', type=int, help='number of rows in data matrix')
    parser.add_argument(
        'n', metavar='cols', type=int, help='number of variables')
    parser.add_argument(
        '-d', '--density', type=float, default=.01, help='density of data '
        'matrix A')
    args = parser.parse_args()

    path = check_path(args.output)
    lp = random_linear_program(m=args.m, n=args.n, density=args.density)
    save_problem(path, lp)
        

if __name__ == '__main__':
    main()
