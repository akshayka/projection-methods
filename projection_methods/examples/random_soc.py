import projection_methods.cones as cones
from projection_methods.algorithms.alternating_projections import AlternatingProjections
from projection_methods.algorithms.qp_solver import QPSolver

import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cones', nargs='+', help='list of cones',
    default=[cones.ConeTypes.REALS, cones.ConeTypes.SOC])
parser.add_argument('-d', '--dims', nargs='+', help='dimensions of cones',
    default=[500, 1000])
parser.add_argument('-a', '--affine_dim', type=int, help='dimension of matrix',
    default=1000)
args = vars(parser.parse_args())

cone = cones.Cone(types=args['cones'], dims=args['dims'])
fp, data = cones.random_feasibility_problem(cone, affine_dim=args['affine_dim'])
print 'Solving with alternating projections ...'
ap = AlternatingProjections()
ap_iterates = ap.solve(fp)
print 'Solving with QP ...'
qp = QPSolver()
qp_iterates = qp.solve(fp)

print 'Min error for alternating projections %f ' % min(ap.errors)
print 'Min error for qp solver %f ' % min(qp.errors)

plt.figure()
plt.plot(ap.errors, label='alternating projections')
plt.plot(qp.errors, label='QP Solver')
plt.legend()
plt.show()
