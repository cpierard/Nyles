import matplotlib.pyplot as plt
import numpy as np
import diagnostics as diag
import numpy.ma as ma
import tracemalloc

exp = diag.plume('../../data/Nyles/forced_plume_32z/', 'forced_plume_32z')
#exp = diag.plume('../../data/Nyles/plume_2days_pressure/', 'plume_2days_pressure')

tracemalloc.start()
b_budget = exp.Budget('b', 0.25, 0.55)
current, peak = tracemalloc.get_traced_memory()
print(f"Budget memory usage {current / 10**6}MB; Peak was {peak / 10**6}MB")
tracemalloc.stop()

tracemalloc.start()
b_budget2 = exp.Budget_2('b', 0.25, 0.55)
current, peak = tracemalloc.get_traced_memory()
print(f"Budget_2 memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
tracemalloc.stop()
