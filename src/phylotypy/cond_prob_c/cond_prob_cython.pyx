# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
import tempfile
import os


def calc_genus_conditional_prob(
        cnp.ndarray[cnp.int32_t, ndim=1] kmer_indices,
        cnp.ndarray[cnp.int32_t, ndim=1] kmer_offsets,
        cnp.ndarray genera_idx_input,
        cnp.ndarray word_specific_priors,
) -> cnp.ndarray:
    cdef:
        Py_ssize_t seq_idx, k
        int n_kmers = word_specific_priors.shape[0]
        int n_sequences = genera_idx_input.shape[0]
        int n_genera, genus, start, end
        cnp.ndarray[cnp.int32_t, ndim=1] genera_idx
        cnp.ndarray[cnp.float32_t, ndim=1] priors
        cnp.ndarray[cnp.float32_t, ndim=1] genus_counts_f
        cnp.ndarray[cnp.float32_t, ndim=2] genus_count
        
    if genera_idx_input.dtype == np.int32:
        genera_idx = genera_idx_input
    else:
        genera_idx = genera_idx_input.astype(np.int32)
        
    if word_specific_priors.dtype == np.float32:
        priors = word_specific_priors
    else:
        priors = word_specific_priors.astype(np.float32)
    
    n_genera = int(genera_idx.max()) + 1
    genus_counts_f = np.bincount(genera_idx, minlength=n_genera).astype(np.float32)

    # Back to simple in-memory allocation
    genus_count = np.zeros((n_kmers, n_genera), dtype=np.float32)

    for seq_idx in range(n_sequences):
        genus = genera_idx[seq_idx]
        start = kmer_offsets[seq_idx]
        end = kmer_offsets[seq_idx + 1]
        for k in range(start, end):
            genus_count[kmer_indices[k], genus] += 1.0

    # In-place ops — no copies
    genus_counts_f += 1.0
    genus_count += priors.reshape(-1, 1)
    genus_count /= genus_counts_f
    np.log(genus_count, out=genus_count)

    return genus_count


def calc_genus_conditional_prob_mmap(
        cnp.ndarray[cnp.int32_t, ndim=1] kmer_indices,
        cnp.ndarray[cnp.int32_t, ndim=1] kmer_offsets,
        cnp.ndarray genera_idx_input,
        cnp.ndarray word_specific_priors,
) -> cnp.ndarray:
    cdef:
        Py_ssize_t seq_idx, genus_idx, kmer_idx
        int n_kmers = word_specific_priors.shape[0]
        int n_genera
        int c_start, c_end, genus, start, end
        int chunk_size = 10000
        int n_sequences = genera_idx_input.shape[0]
        cnp.ndarray[cnp.int32_t, ndim=1] genera_idx
        cnp.ndarray[cnp.float32_t, ndim=1] priors
        cnp.ndarray[cnp.float32_t, ndim=1] genus_counts_f
        
    if genera_idx_input.dtype == np.int32:
        genera_idx = genera_idx_input
    else:
        genera_idx = genera_idx_input.astype(np.int32)
        
    if word_specific_priors.dtype == np.float32:
        priors = word_specific_priors
    else:
        priors = word_specific_priors.astype(np.float32)
        
    n_genera = int(genera_idx.max()) + 1
    
    # genus_counts per genus (needed for normalization)
    genus_counts_f = np.bincount(genera_idx,
                                 minlength=n_genera).astype(np.float32)
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mmap")
    tmp.close()
    
    # big memory allocation
    genus_count = np.memmap(tmp.name, dtype=np.float32,
                            mode="w+", shape=(n_kmers, n_genera))
    
    try:
        # Main computation loop
        for c_start in range(0, n_sequences, chunk_size):
            c_end = min(c_start + chunk_size, n_sequences)
            for seq_idx in range(c_start, c_end):
                genus = genera_idx[seq_idx]
                start = kmer_offsets[seq_idx]
                end = kmer_offsets[seq_idx + 1]
                for k in range(start, end):
                    genus_count[kmer_indices[k], genus] += 1.0
            genus_count.flush()
            
        genus_counts_f += 1.0
        genus_count += priors.reshape(-1,1)
        genus_count /= genus_counts_f
        np.log(genus_count, out=genus_count)
        genus_count.flush()
        result = np.array(genus_count)
    finally:
        del genus_count
        os.unlink(tmp.name)
        
    return result
