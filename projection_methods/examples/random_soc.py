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
import numpy as np
import pprint
import tabulate
import pdb


def solve(solver, problem, iters, label, table_data, dist_fig, delta_fig):
    opt = solver.solve(problem)
    dists = []
    logging.info('Calculating errors ...')
    iters = [i for i in iters if i < len(solver.iterates)]
    for i in iters:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--load_cached_problem', type=str,
        help='name of cached file to create or load',
        default=None)
    parser.add_argument(
        '-p', '--plot_file_pfx', type=str, required=True,
        help='prefix of plot files to be generated')
    parser.add_argument(
        '-c', '--cones', nargs='+',
        default=[cones.ConeTypes.REALS, cones.ConeTypes.SOC],
        help='list of cones')
    parser.add_argument(
        '-d', '--dims', nargs='+', default=[500, 1000],
        help='dimensions of cones') 
    parser.add_argument(
        '-a', '--affine_dim', type=int, default=1000,
        help='dimension of matrix')
    parser.add_argument(
        '-ap', action='store_true', help='whether to run the AP solver')
    parser.add_argument(
        '-qp', action='store_true', help='whether to run the QP solver')
    parser.add_argument(
        '-dk', action='store_true', help='whether to run the Dykstra solver')
    parser.add_argument(
        '-i', '--max_iters', type=int, default=100,
        help='maximum number of iterations to run each algorithm')
    parser.add_argument(
        '-s', '--step', type=int, default=10,
        help='iteration step to use when tabulating, plotting')
    parser.add_argument(
        '-mo', '--momentum', nargs=2, default=None,
        help='alpha and beta values for momentum, defaults to no momentum.')
    parser.add_argument(
        '-dp', '--discard_policy', type=str, default='evict',
        help='discard policy for the QP solver')
    parser.add_argument(
        '-ip', '--include_probability', nargs='+', type=float,
        help='probability with which to include a halfpsace in the QP solver',
        default=[1.0])
    parser.add_argument(
        '-m', '--memory', nargs='+', type=int, default=[2],
        help='list of memory lengths for qp')
    parser.add_argument(
        '-ll', '--log_level', type=str, default='INFO',
        help='logging level (see the logging module for a list of valid levels')

    args = vars(parser.parse_args())
    logging.basicConfig(
        format='[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s',
        level=eval('logging.%s' % args['log_level']))
    if not args['qp'] and not args['ap'] and not args['dk']:
        raise ValueError('At least one of ap, qp, and dk must be specified.')

    # Problem set-up
    if args['momentum'] is not None:
        momentum = {
            'alpha' : args['momentum'][0],
            'beta' : args['momentum'][1]
        }
        momentum_sfx = '(alpha: %f, beta: %f)' % (
            args['momentum'][0], args['momentum'][1])

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
        logging.info('Solving with alternating projections ...')
        ap = AlternatingProjections(
            max_iters=args['max_iters'], initial_point=initial_point)
        solve(ap, fp, iters, 'AP', table_data, dist_fig, delta_fig)

        if args['momentum'] is not None:
            logging.info('Solving with alternating projections + momentum ...')
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
                if args['momentum'] is not None:
                    logging.info( 
                        'Solving with QP, mem: %d, ip: %.2f, + momentum ...',
                        m, ip)
                    qp.momentum = args['momentum']
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
    plt.title('Distances from the intersection')
    plt.legend()
    plt.savefig(args['plot_file_pfx'] + '_dists.png')

    plt.figure(delta_fig)
    plt.title('Deltas between iterates')
    plt.legend()
    plt.savefig(args['plot_file_pfx'] + '_deltas.png')

if __name__ == '__main__':
    main()
