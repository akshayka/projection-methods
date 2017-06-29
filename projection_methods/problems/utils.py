import cPickle
from pathlib2 import PosixPath
import sys

def die_if(cond, msg):
    if cond:
        print 'Error: ' + msg
        sys.exit(1)


def check_path(path):
    path = PosixPath(path).expanduser()
    die_if(path.is_dir(), 'Please enter a filename, not a directory.')
    die_if(not path.parents[0].is_dir(), 'You are trying to save your '
        'problem in a non-extant directory.')
    die_if(path.is_file(), 'You are trying to overwrite an extant file; '
        'this is not allowed.')
    return path


# Note that the supplied path should be a pathlib2.Path instance
def save_problem(posix_path, problem):
    with posix_path.open('wb') as f:
        cPickle.dump(problem, f, protocol=cPickle.HIGHEST_PROTOCOL)
    with open(str(posix_path) + '.txt', 'wb') as f:
        f.write(str(problem))
    print 'Saved problem at ' + str(posix_path)
        

