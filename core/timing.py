from functools import wraps
from time import time
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
stats = {}

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        if 'timing' in kw.keys():
            pass
        else:
            name = f.__name__
            if name in stats.keys():
                stats[name] += [te-ts]
            else:
                stats[name] = [te-ts]
        #print('func:%r took: %2.4e sec' % (f.__name__,  te-ts))
        return result
    return wrap

def write_timings(path):
    fid = open('%s/timing.pkl' % path, 'bw')
    pickle.dump(stats, fid)
    
def analyze_timing(path):

    mpl.rcParams['font.size'] = 14
    mpl.rcParams['lines.linewidth'] = 2

    filename = '%s/timing.pkl' % path
    pngtiming = '%s/timing.png' % path

    f = open(filename, 'br')
    timing = pickle.load(f)

    mean = []
    keys = []
    for k, vals in timing.items():
        mean += [np.mean(vals)]
        keys += [k]

    idx = np.argsort(mean)

    plt.figure(figsize=(10, 5))
    for k in idx[::-1]:
        vals = timing[keys[k]]
        plt.loglog(vals, label=keys[k])

    plt.xlabel('iteration')
    plt.ylabel('time [s]')
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig(pngtiming)
