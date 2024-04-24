import numba
import numpy as np
import numpy.typing as npt


@numba.njit
def count1D(counts: npt.NDArray, xx: npt.NDArray) -> None:
    assert len(counts.shape) == 1
    for i in range(len(xx)):
        counts[xx[i]] += 1


def mkindex(indexed) -> tuple[npt.NDArray, npt.NDArray, np.uint32]:
    last = np.uint64(np.max(indexed) + 1)
    counts = np.zeros(last, dtype=np.uint64)
    count1D(counts, indexed)
    index = np.empty(shape=(counts.shape[0] + 1,), dtype=np.uint64)
    index[0] = 0
    np.cumsum(counts, out=index[1:])
    return index, counts, last
