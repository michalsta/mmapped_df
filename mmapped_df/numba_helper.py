import numba
import numpy as np

@numba.njit(boundscheck=True)
def mkindex(indexed):
    index = np.empty(shape=indexed[-1]+2, dtype=np.uint64)
    idx = 0
    for ii in range(len(index)-1):
        while(indexed[idx]) < ii:
            idx += 1
        index[ii] = idx
    index[-1] = len(indexed)+1
    return index