import numba
import numpy as np
import numpy.typing as npt


@numba.njit
def count1D(counts: npt.NDArray, xx: npt.NDArray) -> None:
    assert len(counts.shape) == 1
    for i in range(len(xx)):
        counts[xx[i]] += 1


def mkindex(mkindex) -> tuple[npt.NDArray, npt.NDArray, np.uint32]:
    last = mkindex[-1]
    counts = np.zeros(last + 1, dtype=np.uint32)
    count1D(counts, mkindex)
    index = np.empty(shape=(counts.shape[0] + 1,), dtype=np.uint64)
    np.cumsum(counts, out=index[1:])
    return index, counts, last


# @numba.njit(boundscheck=True)
# def mkindex(indexed):
#     index = np.empty(shape=indexed[-1] + 2, dtype=np.uint64)
#     idx = 0
#     for ii in range(len(index) - 1):
#         while (indexed[idx]) < ii:
#             idx += 1
#         index[ii] = idx
#     index[-1] = len(indexed) + 1
#     return index
