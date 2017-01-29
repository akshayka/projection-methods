import projection_methods.algorithms.utils as utils
import projection_methods.cones as cones
from projection_methods.algorithms.alternating_projections import (
    AlternatingProjections
)
from projection_methods.algorithms.dykstra import Dykstra
from projection_methods.algorithms.qp_solver import QPSolver

import argparse
import cPickle
import itertools
import logging
import os
import matplotlib.pyplot as plt
from mpldatacursor import datacursor
import numpy as np
import pprint
import tabulate


def solve(solver, problem, iters, label, table_data, dist_fig, delta_fig):
    opt = solver.solve(problem)
    dists = []
    logging.info('Calculating errors ...')
    iters = [i for i in iters if i < len(solver.iterates)]
    for i in iters:
        # TODO(akshayka): Reevaluate the definition of error here.
        # Assuming that the sequence of itereates converged, it might
        # be informative to keep track of the distance from the final iterate
        # as well.
        _, dist = utils.project_aux(
            solver.iterates[i], problem.cvx_sets[0] + problem.cvx_sets[1],
            problem.cvx_var)
        dists.append(dist)
    if not np.all(np.diff(dists) <= 0):
        logging.warning(
            'Error sequence for %s is not decreasing; # violations: %d',
            label, np.sum((np.diff(dists) >= 0)))
    table_data.append([label] + dists)
    plt.figure(dist_fig)
    plt.plot(iters, dists, '-o', label=label)
    plt.figure(delta_fig)
    plt.plot(
        range(len(solver.iterates) - 1),
        [np.linalg.norm(v - w, 2) for v, w in itertools.izip(
            solver.iterates[:-1], solver.iterates[1:])],
        '-o', label=label)


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
        '-l', '--load_cached_problem', type=str, default=None,
        help='name of cached file to create or load')
    parser.add_argument(
        '-p', '--plot_file_pfx', type=str, default=None,
        help='prefix of plot files to be generated')
    parser.add_argument(
        '-ll', '--log_level', type=str, default='INFO',
        help='logging level (see the logging module for a list of valid levels')
    # --- experiment parameters --- #
    parser.add_argument(
        '-i', '--max_iters', type=int, default=100,
        help='maximum number of iterations to run each algorithm')
    parser.add_argument(
        '-s', '--step', type=int, default=10,
        help='iteration step to use when tabulating, plotting')
    # --- problem parameters --- #
    parser.add_argument(
        '-c', '--cones', nargs='+',
        default=[cones.ConeTypes.REALS, cones.ConeTypes.SOC],
        help='list of cones')
    parser.add_argument(
        '-d', '--dims', nargs='+', type=int, default=[500, 1000],
        help='dimensions of cones')
    parser.add_argument(
        '-a', '--affine_dim', type=int, default=1000,
        help='dimension of matrix')
    # --- the algorithms --- #
    parser.add_argument(
        '-ap', action='store_true', help='whether to run the AP solver')
    parser.add_argument(
        '-qp', action='store_true', help='whether to run the QP solver')
    parser.add_argument(
        '-dk', action='store_true', help='whether to run the Dykstra solver')
    # --- options for the AP solver --- #
    parser.add_argument(
        '-apmo', '--ap_momentum', nargs='+', type=float, default=None,
        help='sequence of alpha and beta values for AP momentum, '\
             'defaults to no momentum; for example -- 0.8 0.2 0.9 0.1.'\
             'would produce two pairs of alpha-beta values: '\
             '(0.8, 0.2) and (0.9, 0.1)')
    parser.add_argument(
        '-psa', '--plane_search_affine', nargs='+', type=int, default=[1],
        help='# iterates to include when performing a plane search on the '\
             'affine set; 1 ==> no plane search')
    parser.add_argument(
        '-psc', '--plane_search_cone', nargs='+', type=int, default=[1],
        help='# iterates to include when performing a plane search on the '\
             'convex cone; 1 ==> no plane search')
    # --- options for the QP solver --- #
    parser.add_argument(
        '-qpmo', '--qp_momentum', nargs='+', type=float, default=None,
        help='sequence of alpha and beta values for AP momentum, '\
             'defaults to no momentum; for example -- 0.8 0.2 0.9 0.1.'\
             'would produce two pairs of alpha-beta values: '\
             '(0.8, 0.2) and (0.9, 0.1)')
    parser.add_argument(
        '-dp', '--discard_policy', type=str, default='evict',
        help='discard policy for the QP solver')
    parser.add_argument(
        '-ip', '--include_probability', nargs='+', type=float, default=[1.0],
        help='probability with which to include a halfpsace in the QP solver')
    parser.add_argument(
        '-m', '--memory', nargs='+', type=int, default=[2],
        help='list of memory lengths for qp')

    args = vars(parser.parse_args())
    logging.basicConfig(
        format='[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s',
        level=eval('logging.%s' % args['log_level']))
    if not args['qp'] and not args['ap'] and not args['dk']:
        raise ValueError('At least one of ap, qp, and dk must be specified.')

    if args['ap_momentum'] is not None and len(args['ap_momentum']) % 2 != 0:
        raise ValueError('Length of momentum must be a multiple of 2.')

    if args['qp_momentum'] is not None and len(args['qp_momentum']) % 2 != 0:
        raise ValueError('Length of momentum must be a multiple of 2.')

    # Problem set-up
    if (args['load_cached_problem'] is not None
            and os.path.isfile(args['load_cached_problem'])):
        logging.info(
            'loading cached problem %s ...', args['load_cached_problem'])
        pkl_file = open(args['load_cached_problem'], 'rb')
        cone, data, initial_point = cPickle.load(pkl_file)
        fp = cones.make_feasibility_problem(cone, sum(cone.dims), data)
        pkl_file.close()
    else:
        cone = cones.Cone(types=args['cones'], dims=args['dims'])
        fp, data = cones.random_feasibility_problem(
            cone, affine_dim=args['affine_dim'])
        initial_point = np.random.randn(fp.var_dim, 1)
        if args['load_cached_problem'] is not None:
            logging.info(
                'caching problem as %s ...', args['load_cached_problem'])
            with open(args['load_cached_problem'], 'wb') as pkl_file:
                cPickle.dump((cone, data, initial_point), pkl_file)

    table_data = []
    dist_fig = 0
    delta_fig = 1

    iters = range(0, args['max_iters'] + 1, args['step'])
    if args['max_iters'] not in iters:
        iters = iters + [args['max_iters']]

    # evaluate AP
    if args['ap']:
        ap = AlternatingProjections(
            max_iters=args['max_iters'], initial_point=initial_point)
        solve(ap, fp, iters, 'AP', table_data, dist_fig, delta_fig)

        for psa in args['plane_search_affine']:
            for psc in args['plane_search_cone']:
                if psa == 1 and psc == 1:
                    continue
                logging.info('Solving with alternating projections + '\
                    'plane search (%d, %d) ...' % (psa, psc))
                ap.plane_search[0] = psa
                ap.plane_search[1] = psc
                solve(
                    ap, fp, iters, 'AP + plane search (%d, %d)' % (psa, psc),
                    table_data, dist_fig, delta_fig)

        if args['ap_momentum'] is not None:
            for i in xrange(0, len(args['ap_momentum']), 2):
                momentum, momentum_sfx = momentum_fmt(args['ap_momentum'], i)
                logging.info(
                    'Solving with alternating projections + momentum ...')
                ap.momentum = momentum
                solve(
                    ap, fp, iters, 'AP + momentum ' + momentum_sfx, table_data,
                    dist_fig, delta_fig)

    # evaluate QP
    if args['qp']:
        qp = QPSolver(
            max_iters=args['max_iters'], initial_point=initial_point)
        qp.discard_policy = args['discard_policy']
        for m in args['memory']:
            for ip in args['include_probability']:
                logging.info('Solving with QP, mem: %d, ip: %.2f ...', m, ip)
                qp.include_probability = ip
                qp.memory = m
                solve(
                    qp, fp, iters,
                    'QP, mem (%d), ip (%.2f)' % (m, ip), table_data,
                    dist_fig, delta_fig)
                if args['qp_momentum'] is not None:
                    for i in xrange(0, len(args['qp_momentum']), 2):
                        momentum, momentum_sfx = momentum_fmt(
                            args['qp_momentum'], i)
                        logging.info(
                            'Solving with QP, mem: %d, ip: %.2f + momentum ...',
                            m, ip)
                        qp.momentum = momentum
                        solve(
                            qp, fp, iters,
                            'QP, mem (%d), ip (%.2f), %s' % (m, ip, momentum_sfx),
                            table_data, dist_fig, delta_fig)
                        qp.momentum = None

    # evaluate Dykstra's
    if args['dk']:
        dk = Dykstra(max_iters=args['max_iters'], initial_point=initial_point)
        solve(dk, fp, iters, 'Dykstra', table_data, dist_fig, delta_fig)

    print '+-------------------------------+'
    print '              args               '
    print '+-------------------------------+'
    pprint.pprint(args)
    print '+-------------------------------+'
    print '             results             '
    print '+-------------------------------+'
    headers = ['algorithm'] + ['error @ iter %d' % i for i in iters]
    print tabulate.tabulate(
        table_data, headers=headers, tablefmt='grid')

    plt.figure(dist_fig)
    plt.title('Distance from the intersection')
    plt.legend()
    plt.yscale('log')
    if args['plot_file_pfx'] is not None:
        plt.savefig(args['plot_file_pfx'] + '_dists.png')
    else:
        datacursor(formatter='{label}'.format)
        plt.show(dist_fig)

    plt.figure(delta_fig)
    plt.title('Deltas between iterates')
    plt.legend()
    plt.yscale('log')
    if args['plot_file_pfx'] is not None:
        plt.savefig(args['plot_file_pfx'] + '_deltas.png')
    else:
        plt.show(delta_fig)

if __name__ == '__main__':
    main()
