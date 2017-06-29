import argparse
import cPickle
import logging
import os
import numpy as np
import time

from projection_methods.algorithms.altp import AltP
from projection_methods.algorithms.avgp import AvgP
from projection_methods.algorithms.apop import APOP
from projection_methods.algorithms.dykstra import Dykstra
from projection_methods.algorithms.scs_admm import SCSADMM
from projection_methods.oracles.affine_set import AffineSet
from projection_methods.oracles.dynamic_polyhedron import PolyOuter
from projection_methods.problems.problems import SCSProblem


k_alt_p = 'altp'
k_avg_p = 'avgp'
k_apop = 'apop'
k_dykstra = 'dyk'
k_scs= 'scs'
k_solvers = frozenset([k_alt_p, k_avg_p, k_apop, k_dykstra, k_scs])

k_exact = 'exact'
k_elra = 'elra'
k_erandom = 'erandom'
k_subsample = 'subsample'
k_outers = {
    k_exact: PolyOuter.EXACT,
    k_elra: PolyOuter.ELRA,
    k_erandom: PolyOuter.ERANDOM,
    k_subsample: PolyOuter.SUBSAMPLE,
}


def main():
    example =\
    """example usage:
    python experiment.py problems/sv/convex_affine/1000_square.pkl
    results/convex_affine/1000_square/apop apop -n apop_exact -o exact"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=example)
    # --- input/output --- #
    parser.add_argument(
        'problem', metavar='P', type=str, default=None,
        help='path to problem on which to experiment')
    parser.add_argument(
        'output', metavar='O', type=str, default=None,
        help=('output path specifying location in which to save results; '
        'note that a timestamp will be appended to the path'))
    parser.add_argument(
        '-n', '--name', type=str, default='',
        help='descriptive name of experiment')
    parser.add_argument(
        '-ll', '--log_level', type=str, default='INFO',
        help='logging level (see the logging module for a list of levels)')
    parser.add_argument(
        '-v', '--verbose', action='store_true')
    # --- algorithms --- #
    parser.add_argument(
        'solver', metavar='S', type=str, default='apop',
        help='which solver to use; one of ' + str(list(k_solvers)))
    # --- options for k_apop --- #
    parser.add_argument(
        '-alt', action='store_true', help=('use the alternating method '
        'instead of the averaging one for k_apop'))
    parser.add_argument(
        '-o', '--outer', type=str, default='exact',
        help=('outer approximation management policy for k_apop; one of ' +
        str(k_outers.keys())))
    parser.add_argument(
        '-mhyp', '--max_hyperplanes', type=int, default=None,
        help=('maximum number of hyperplanes allowed in the outer approx; '
        'defaults to unlimited.'))
    parser.add_argument(
        '-mhlf', '--max_halfspaces', type=int, default=None,
        help=('maximum number of halfspaces allowed in the outer approx; '
        'defaults to unlimited.'))
    parser.add_argument(
        '-t', '--theta', type=float, default=1.0,
        help=('over/under-relaxation: must be in (0, 2)'))
    # --- options shared by at least two solvers --- #
    parser.add_argument(
        '-i', '--max_iters', type=int, default=100,
        help='maximum number of iterations to run the algorithm')
    parser.add_argument(
        '-a', '--do_all_iters', action='store_true',
        help='perform exactly max_iters iterations.')
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
    if args['solver'] not in k_solvers:
        raise ValueError('Invalid solver choice %s' % args['solver'])

    # Load in the problem
    logging.info(
        'loading cached problem %s ...', args['problem'])
    with open(args['problem'], 'rb') as pkl_file:
        problem = cPickle.load(pkl_file)
    
    fn = '_'.join([args['output'], time.strftime("%Y%m%d-%H%M%S")]) + '.pkl'
    if not os.access(os.path.dirname(fn), os.W_OK):
        raise ValueError('Invalid output path %s' % fn)

    if args['solver'] == k_alt_p:
        solver = AltP(max_iters=args['max_iters'], atol=args['atol'],
            do_all_iters=args['do_all_iters'], momentum=args['momentum'],
            verbose=args['verbose'])
    elif args['solver'] == k_avg_p:
        solver = AvgP(max_iters=args['max_iters'], atol=args['atol'],
            momentum=args['momentum'], verbose=args['verbose'])
    elif args['solver'] == k_apop:
        solver = APOP(max_iters=args['max_iters'], atol=args['atol'],
            do_all_iters=args['do_all_iters'],
            outer_policy=k_outers[args['outer']],
            max_hyperplanes=args['max_hyperplanes'],
            max_halfspaces=args['max_halfspaces'],
            momentum=args['momentum'],
            average=not args['alt'],
            theta=args['theta'],
            verbose=args['verbose'])
    elif args['solver'] == k_dykstra:
        solver = Dykstra(max_iters=args['max_iters'], atol=args['atol'],
            do_all_iters=args['do_all_iters'], verbose=args['verbose'])
    elif args['solver'] == k_scs:
        solver = SCSADMM(max_iters=args['max_iters'], atol=args['atol'],
            do_all_iters=args['do_all_iters'], verbose=args['verbose'])
    else:
        raise ValueError('Invalid solver choice %s' % args['solver'])

    it, res, status = solver.solve(problem)
    name = args['name'] if len(args['name']) > 0 else args['solver']
    data = {'it': it, 'res': res, 'status': status,
            'problem': args['problem'], 'name': name, 'solver': args['solver']}

    if isinstance(problem, SCSProblem):
        data['kappa'] = problem.kappa(it[-1])
        data['tau'] = problem.tau(it[-1])
        if data['tau'] > -1e-6 and np.isclose(data['kappa'], 0, atol=1e-4):
            data['case'] = 'primal_dual_optimal'
        elif np.isclose(data['tau'], 0, atol=1e-4) and data['kappa'] > 1e-6:
            data['case'] = 'infeasible'
        else:
            data['case'] = 'indeterminate'
        data['obj_val'] = (problem.objective_value(problem.p(it[-1])) /
            data['tau']) if data['tau'] != 0 else float('inf')
        data['opt_val'] = problem.optimal_value()
        data['primal_res'] = data['obj_val'] - data['opt_val']
        data['rel_error'] = abs(data['primal_res']) / abs(data['opt_val'])

    with open(fn, 'wb') as f:
        cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
    last_res = sum(res[-1]) if hasattr(res[-1], '__iter__') else res[-1]
    print '%s terminated after %d iterations; last residual %.5e' % (
        name, len(it), last_res)


if __name__ == '__main__':
    main()
