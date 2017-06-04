import argparse
import cPickle
import logging
import time

from projection_methods.algorithms.altp import AltP
from projection_methods.algorithms.apop import APOP


ALTP = 'altp'
APOP = 'apop'
SOLVERS = frozenset([ALTP, APOP])


def momentum_fmt(momentum_list, idx):
    momentum = {
        'alpha' : momentum_list[idx],
        'beta' : momentum_list[idx+1]
    }
    momentum_sfx = '(alpha: %.2f, beta: %.2f)' % (
        momentum['alpha'], momentum['beta'])
    return momentum, momentum_sfx


def main():
    parser = argparse.ArgumentParser()
    # --- input/output --- #
    parser.add_argument(
        'problem', metavar='P', type=str, default=None,
        help='path to problem on which to experiment')
    parser.add_argument(
        'output', metavar='O', type=str, default=None,
        help=('output path specifying location in which to save results; '
        'note that a timestamp and solver name will be appended to the path')
    parser.add_argument(
        '-ll', '--log_level', type=str, default='INFO',
        help='logging level (see the logging module for a list of valid levels')
    # --- algorithms --- #
    parser.add_argument(
        'solver', metavar='S', type=str, default='apop',
        help='which solver to use; one of ' + str(SOLVERS))
    # --- options for APOP --- #
    parser.add_argument(
        '-alt', action='store_false', help=('use the alternating method '
        'instead of the averaging one for APOP'))
    parser.add_argument(
        '-o', '--outer', type=str, default='exact',
        help='outer approx. management policy for APOP')
    parser.add_argument(
        '-mhyp', '--max_hyperplanes', type=int, default=None,
        help=('maximum number of hyperplanes allowed in the outer approx; '
        'defaults to unlimited.')
    parser.add_argument(
        '-mhlf', '--max_halfspaces', type=int, default=None,
        help=('maximum number of halfspaces allowed in the outer approx; '
        'defaults to unlimited.')
    # --- options shared by at least two solvers --- #
    parser.add_argument(
        '-i', '--max_iters', type=int, default=100,
        help='maximum number of iterations to run the algorithm')
    parser.add_argument(
        '-mo', '--momentum', nargs=2, type=float, default=None,
        help=('alpha and beta values for momentum (defaults to no momentum); '
        'e.g.: 0.95 0.05 yields alpha == 0.95, beta == 0.05'))
    parser.add_argument(
        '-atol', type=float, default=1e-4,
        help='residual threshold for optimality')

    args = vars(parser.parse_args())
    logging.basicConfig(
        format='[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s',
        level=eval('logging.%s' % args['log_level']))
    if not args['solver'] not in SOLVERS:
        raise ValueError('Invalid solver choice %s' % args['solver'])

    # Load in the problem
    logging.info(
        'loading cached problem %s ...', args['problem'])
    with open(args['problem'], 'rb') as pkl_file:
        problem = cPickle.load(pkl_file)
    

    if args['solver'] == ALTP:
        solver = AltP(max_iters=args['max_iters'], atol=args['atol'],
            momentum=args['momentum'])
    elif args['solver'] == APOP:
        solver = APOP(max_iters=args['max_iters'], atol=args['atol'],
            outer_policy=OUTER_MAP[args['outer']],
            max_hyperplanes=args['max_hyperplanes'],
            max_halfspaces=args['max_halfspaces'],
            momentum=args['momentum'],
            average=not args['alt'])
    else:
        raise ValueError('Invalid solver choice %s' % args['solver'])

    it, res, status = solver.solve(problem)
    data = {'it': it, 'res': res, 'status': status}
    fn = '_'.join(args['output'],
        time.strftime("%Y%m%d-%H%M%S"),
        args['solver']) + '.pkl'
    with open fn as f:
        cPickle.dump(data, f)

if __name__ == '__main__':
    main()
