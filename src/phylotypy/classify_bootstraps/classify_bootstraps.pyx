# File: classify_bootstraps.pyx

import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport cython
from libc.stdlib cimport malloc, free

# Define C types for better performance
ctypedef np.int64_t INT64_t
ctypedef np.float64_t FLOAT64_t  # Adjust if your conditional_prob has a different dtype
ctypedef np.float32_t FLOAT32_t

@cython.boundscheck(False)  # Turn off bounds-checking
@cython.wraparound(False)  # Turn off negative index wrapping
@cython.cdivision(True)  # Disable division by zero checks
def classify_bootstraps_cython(np.ndarray[INT64_t, ndim=2] bs_indices,
                               np.ndarray[FLOAT32_t, ndim=2] conditional_prob):
    """
    Classify bootstrapped samples using conditional probabilities.

    Parameters:
    -----------
    bs_indices : 2D array of int64
        Indices for bootstrapped samples
    conditional_prob : 2D array of float64
        Conditional probability matrix

    Returns:
    --------
    classifications : 1D array of int64
        Classification results for each bootstrap sample
    """
    cdef Py_ssize_t n_bootstraps = bs_indices.shape[0]
    cdef Py_ssize_t n_samples = bs_indices.shape[1]
    cdef Py_ssize_t n_classes = conditional_prob.shape[1]

    # Create output array
    cdef np.ndarray[INT64_t, ndim=1] classifications = np.empty(n_bootstraps, dtype=np.int64)

    # Declare variables for loops
    cdef Py_ssize_t i, j, k, max_idx
    cdef FLOAT64_t max_val, current_val
    cdef FLOAT64_t *sums  # C array for sums
    cdef INT64_t idx

    # Parallel execution with proper GIL handling
    with nogil:
        for i in prange(n_bootstraps):
            # Allocate memory for sums array (using C malloc instead of np.zeros)
            sums = <FLOAT64_t *> malloc(n_classes * sizeof(FLOAT64_t))

            # Initialize sums to zero
            for k in range(n_classes):
                sums[k] = 0.0

            # Sum conditional probabilities for each class
            for j in range(n_samples):
                idx = bs_indices[i, j]
                for k in range(n_classes):
                    sums[k] += conditional_prob[idx, k]

            # Find the class with highest sum
            max_val = sums[0]
            max_idx = 0
            for k in range(1, n_classes):
                if sums[k] > max_val:
                    max_val = sums[k]
                    max_idx = k

            # Store result
            classifications[i] = max_idx

            # Free allocated memory
            free(sums)

    return classifications
