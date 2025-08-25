cimport cython
import numpy as np
import scipy.stats as ss


ctypedef fused indices_type:
    int
    long

ctypedef fused indptr_type:
    int
    long


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple csr_to_csc(
    const float[:] input_data,
    indices_type[:] input_indices,
    indptr_type[:] input_indptr,
    int M,
    int N,
    const long[:] ords
):
    """ This routine in addition group cells by clusters (ords)"""
    cdef Py_ssize_t i, j, pos, col

    output_indptr = np.zeros(N+1, dtype=np.int64)
    cdef long[:] indptr = output_indptr
    cdef long[:] counter = np.zeros(N, dtype=np.int64)

    for i in range(input_indices.size):
        if input_data[i] != 0.0: # in case there are extra 0s in the sparse matrix
            indptr[input_indices[i]+1] += 1
    for i in range(N):
        counter[i] = indptr[i]
        indptr[i + 1] += indptr[i]

    output_data = np.zeros(indptr[N], dtype=np.float32)
    output_indices = np.zeros(indptr[N], dtype=np.int64)
    cdef float[:] data = output_data
    cdef long[:] indices = output_indices

    for i in range(M):
        for j in range(input_indptr[ords[i]], input_indptr[ords[i]+1]):
            if input_data[j] != 0.0:
                col = input_indices[j]
                pos = counter[col]
                data[pos] = input_data[j]
                indices[pos] = i
                counter[col] += 1

    return output_data, output_indices, output_indptr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void partition_indices(const long[:] indices, const long[:] cumsum, long[:] posarr):
    cdef Py_ssize_t i, j, s
    posarr[0] = 0
    i = 0
    s = indices.size
    for j in range(cumsum.size):
        while i < s and indices[i] < cumsum[j]:
            i += 1
        posarr[j + 1] = i


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef float[:, :] calc_auroc(
    int start_pos,
    int end_pos,
    const float[:] data,
    const long[:] indices,
    const long[:] indptr,
    const long[:] n1arr,
    const long[:] n2arr,
    const long[:] cumsum,
    int first_j,
    int second_j,
):
    """
    Calculate AUROC scores for all clusters in cluster_labels, focusing only on genes in [start_pos, end_pos).
    This function assumes the cells are grouped by cluster ids.
    """
    cdef Py_ssize_t ngene, ncluster, nsample
    cdef Py_ssize_t i, j, pos_i
    cdef Py_ssize_t n_nonzero, n_zero
    cdef double avg_zero_rank
    cdef double R1

    ngene = end_pos - start_pos
    ncluster = cumsum.size
    nsample = cumsum[ncluster - 1]

    aurocs_np = np.full((ngene, ncluster), 0.5, dtype=np.float32)
    n1Rs_np = np.zeros(ncluster, dtype=np.float64)
    n1n2_np = np.zeros(ncluster, dtype=np.float64)
    cdef float[:, :] aurocs = aurocs_np
    cdef double[:] n1Rs = n1Rs_np
    cdef double[:] n1n2 = n1n2_np
    cdef long[:] posarr = np.zeros(ncluster + 1, dtype=np.int64)

    for j in range(ncluster):
        n1Rs[j] = (< double > (n1arr[j] + 1)) *n1arr[j] / 2.0
        n1n2[j] = (< double > n1arr[j]) * n2arr[j]

    for i in range(start_pos, end_pos):
        pos_i = i - start_pos
        n_nonzero = indptr[i + 1] - indptr[i]
        n_zero = nsample - n_nonzero
        if n_nonzero != 0:
            ranks = ss.rankdata(data[indptr[i]:indptr[i + 1]]) + n_zero  # np.float64
            _, ties = np.unique(ranks, return_counts=True)
            if n_zero > 0:
                ties = np.concatenate(([n_zero], ties))
            avg_zero_rank = (n_zero + 1.0) / 2.0
            partition_indices(indices[indptr[i]: indptr[i + 1]], cumsum, posarr)
            for j in range(ncluster):
                if n1arr[j] == 0 or n2arr[j] == 0:
                    pass
                elif j == second_j:
                    aurocs[pos_i, j] = 1.0 - aurocs[pos_i, first_j]
                else:
                    R1 = (
                        ranks[posarr[j]:posarr[j + 1]].sum() +
                        (n1arr[j] - (posarr[j + 1] - posarr[j])) *
                        avg_zero_rank
                    )
                    aurocs[pos_i, j] = (R1 - n1Rs[j]) / n1n2[j]

    return aurocs_np
