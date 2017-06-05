import cPickle
from pathlib2 import PosixPath
import sys

import cvxpy

from projection_methods.oracles.reals import Reals
from projection_methods.oracles.zeros import Zeros
import projection_methods.oracles.soc as soc

from projection_methods.oracles.cartesian_product import CartesianProduct
from projection_methods.problems.problem_factory import convex_affine_problem


REALS = 'R'
ZEROS = 'Z'
SEC_ORD_CONE = 'SOC'
SETS = {REALS: Reals, ZEROS: Zeros, SEC_ORD_CONE: soc.SOC}

def die_if(cond, msg):
    if cond:
        print 'Error: ' + msg
        sys.exit(1)

# Read in the output path.
path = PosixPath(raw_input(
    'Please enter an output destination (filename) for your problem: ')
    ).expanduser()
die_if(path.is_dir(), 'Please enter a filename, not a directory.')
die_if(not path.parents[0].is_dir(), 'You are trying to save your '
    'problem in a non-extant directory.')
die_if(path.is_file(), 'You are trying to overwrite an extant file; '
    'this is not allowed.')

# Read in parameters to construct the convex set.
dims = raw_input('Please enter a list of dimensions (e.g., 1 4 2): ')
dims = [int(d) for d in dims.split()]
sets = raw_input('Please enter a list of sets (e.g., R SOC Z): ')
sets = [s for s in sets.split()]
die_if(len(dims) != len(sets), 'length of dims must equal length of sets')

total_dim = sum(dims)
x = cvxpy.Variable(total_dim)
if len(dims) == 1:
    C = SETS[sets[0]](x)
else:
    slices = []
    left = 0
    for d in dims:
        right = left + d
        slices.append(slice(left, right))
        left = right 

    convex_sets = []
    for slx, s in zip(slices, sets):
        convex_sets.append(SETS[s](x[slx]))
    C = CartesianProduct(convex_sets, slices)

# Read in parameters to construct the affine set.
rows = int(raw_input(
    'Please enter number of rows in the data matrix A; e.g., 10: '))
density = float(raw_input('Please enter the desired density of A '
    '(float in (0, 1]): '))

# Construct the problem
problem = convex_affine_problem(C, (rows, total_dim), density)

with path.open('wb') as f:
    cPickle.dump(problem, f, protocol=cPickle.HIGHEST_PROTOCOL)

with open(str(path) + '.txt', 'wb') as f:
    f.write(str(problem))

print 'Saved problem at ' + str(path)
