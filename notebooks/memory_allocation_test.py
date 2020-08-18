import matplotlib.pyplot as plt
import numpy as np
import diagnostics as diag
import numpy.ma as ma
import tracemalloc
import time

#exp = diag.plume('../../data/Nyles/forced_plume_32z/', 'forced_plume_32z')
exp = diag.plume('../../data/Nyles/plume_2days_pressure_4/', 'plume_2days_pressure_4')



start = time.time()
tracemalloc.start()
b_budget = exp.Budget('b', 0.25, 0.55)
current, peak = tracemalloc.get_traced_memory()
print(f"Budget memory usage {current / 10**6}MB; Peak was {peak / 10**6}MB")
tracemalloc.stop()
end = time.time()
print(f'Elapsed time {(end - start)}')
print('----')

start = time.time()
tracemalloc.start()
b_budget2 = exp.Budget_2('b', 0.25, 0.55)
current, peak = tracemalloc.get_traced_memory()
print(f"Budget_2 memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
tracemalloc.stop()
end = time.time()
print(f'Elapsed time {(end - start)}')
