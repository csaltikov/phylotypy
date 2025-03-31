# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp

def calc_genus_conditional_prob(
        list detect_list,
        cnp.ndarray[cnp.int32_t, ndim=1] genera_idx,
        cnp.ndarray[cnp.float32_t, ndim=1] word_specific_priors,
) -> cnp.ndarray:
    cdef:
        Py_ssize_t seq_idx, genus_idx, kmer_idx
        int n_sequences = genera_idx.shape[0]
        int n_genera = np.unique(genera_idx).shape[0]
        int n_kmers = word_specific_priors.shape[0]
        cnp.ndarray[cnp.float32_t, ndim=2] genus_count = np.zeros((n_kmers, n_genera), dtype=np.float32)
        cnp.ndarray[cnp.int32_t, ndim=1] genus_counts = np.unique(genera_idx, return_counts=True)[1].astype(np.int32)
        cnp.ndarray[cnp.float32_t, ndim=2] wi_pi
        list current_detect

    # Main computation loop
    for seq_idx in range(n_sequences):
        genus_idx = genera_idx[seq_idx]
        current_detect = detect_list[seq_idx]

        for kmer_idx in current_detect:
            genus_count[kmer_idx, genus_idx] += 1.0

    # Vectorized NumPy operations
    wi_pi = (genus_count + word_specific_priors.reshape(-1, 1))
    cdef cnp.ndarray[cnp.float32_t, ndim=1] m_1 = (genus_counts.astype(np.float32) + 1)

    cdef cnp.ndarray[cnp.float32_t, ndim=2] divided = np.divide(wi_pi, m_1).astype(np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] genus_cond_prob = np.log(divided).astype(np.float32)

    return genus_cond_prob
