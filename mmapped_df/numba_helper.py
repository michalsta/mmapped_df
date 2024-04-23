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


@numba.njit(parallel=True)
def get_intensity_weighted_mean_mzs(
    ClusterIDs,
    index,
    counts,
    mzs,
    intensities,
):
    means = np.zeros(shape=(len(ClusterIDs),), dtype=np.float64)
    for i in numba.prange(len(ClusterIDs)):
        ClusterID = ClusterIDs[i]
        start = index[ClusterID]
        count = counts[ClusterID]
        means[i] = np.average(
            mzs[start : start + count],
            weights=intensities[start : start + count],
        )
    return means
