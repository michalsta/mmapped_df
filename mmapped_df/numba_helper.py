"""
TODO: make obsolete
"""
from numba_progress import ProgressBar

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


@numba.njit(parallel=True, boundscheck=True)
def count_falses_in_groups(bools: npt.NDArray, index: npt.NDArray) -> npt.NDArray:
    assert len(bools) == index[-1], "Index does not describe bools well."
    counts = np.zeros(dtype=np.uint32, shape=len(index) - 1)
    for i in numba.prange(len(counts)):
        for j in range(index[i], index[i + 1]):
            counts[i] += 1 - bools[j]
    return counts


@numba.njit(parallel=True, boundscheck=True)
def _apply_filter(
    retain_array: npt.NDArray,
    unfiltered_values: npt.NDArray,
    final_location: npt.NDArray,
    index_before_filter: npt.NDArray,
    index_after_filter: npt.NDArray,
    progress_proxy: ProgressBar,
) -> None:
    assert len(unfiltered_values) == index_before_filter[-1]
    for cluster_id in numba.prange(len(index_after_filter) - 1):
        j = index_after_filter[cluster_id]
        for i in range(
            index_before_filter[cluster_id], index_before_filter[cluster_id + 1]
        ):
            if retain_array[i]:
                final_location[j] = unfiltered_values[i]
                j += np.uint32(1)
                progress_proxy.update(1)
