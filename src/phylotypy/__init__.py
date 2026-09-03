import warnings

# Numba's own internal notice about its hashing implementation, unrelated to
# anything in this package's logic. Fires once per process on first JIT
# compilation; safe to silence rather than surface to end users.
warnings.filterwarnings(
    "ignore",
    message="FNV hashing is not implemented in Numba",
    category=UserWarning,
)

from .cond_prob_c import cond_prob_cython
from .classify_bootstraps import classify_bootstraps_cython
from .utilities import read_fasta
from .utilities import utilities
