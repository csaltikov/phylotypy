# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-08-30

### Performance

- `batch_classifier.classify_all()`'s bootstrap-classification step is at least
  2.2x faster (measured on real data). It previously built a one-hot
  `scipy.sparse` matrix per chunk and ran a general sparse-dense matmul to
  score each bootstrap replicate against the reference database — a lot of
  machinery for what's actually a "sum a handful of dense rows and argmax"
  operation. Replaced with a `numba` `@njit(parallel=True)` kernel that does
  exactly that, parallelized across bootstrap replicates. Verified
  byte-identical output against the previous implementation given the same
  bootstrap draw.

### Added

- `classify_sequences()` and `classify_all()` now estimate the memory the
  bootstrap-sampling step will need before allocating it, and raise a clear
  `MemoryError` (with a suggested `num_bs` that would fit) if it would exceed
  30% of the machine's total RAM — this is the scenario most likely to bite
  someone who points classify_sequences() at far more sequences than expected
  (e.g. raw sequencing reads instead of ASVs/OTUs; see the Docs note below).
  The check is dependency-free (reads OS-exposed total memory via
  `os.sysconf`, the same information `top`/Activity Monitor can see; degrades
  to a no-op on platforms where that isn't available, e.g. Windows) and can be
  bypassed with `classify_sequences(..., force=True)` for anyone who knows
  their machine can handle it.

### Fixed

- `classify_all()` never actually passed `kmer_size` down to `bootstrap_all()`,
  which defaults to 8 — so any non-default `kmer_size` silently produced the
  wrong number of bootstrap sub-samples per replicate instead of an error.
  Fixed by threading `kmer_size` through explicitly; found while wiring up the
  memory check above, which needed an accurate value to be meaningful.
- Fixed a latent circular import (`batch_classifier` → `bootstrap` →
  `classifier` → `batch_classifier`) that only surfaced if `batch_classifier`
  happened to be the first `phylotypy` submodule imported directly in a fresh
  process. `bootstrap.py` was importing `classifier` at module level for a
  manual timing comparison inside its own `if __name__ == "__main__":` block —
  the only place it was actually used. Moved to a local import there.

### Changed — breaking

- Removed the `chunk` parameter from `classify_all()`, `ClassifyAll.classify()`,
  and `classify_sequences()`. It existed to bound memory while chunking the
  now-removed `scipy.sparse` step; the numba kernel above has no equivalent
  memory concern, so the parameter no longer did anything.

### Docs

- Clarified in `classify_sequences()` and `read_taxa_fasta()`'s docstrings that
  the sequences being classified should be representative, dereplicated
  sequences (ASVs from DADA2, OTU centroids, or long-read consensus sequences
  for Oxford Nanopore data) — not raw sequencing reads, regardless of
  platform. Explains why: wasted redundant compute, uncorrected sequencing
  errors, and the memory-check scenario above.

### Chore

- Suppressed a known-benign `numba` internal warning ("FNV hashing is not
  implemented in Numba") from both normal library usage and the test suite's
  output — it's informational noise about numba's own hashing implementation,
  unrelated to anything in this package.

## [0.4.0] - 2026-08-30

### Fixed

- `training_data.filter_train_set(n_levels=7)` crashed (`ValueError: Columns must
  be same length as key`) on any reference database containing a taxonomy string
  with more than 7 semicolon-delimited levels — a realistic case for raw,
  uncleaned reference releases (verified against a real raw SILVA release, whose
  taxonomy depths actually span 1 to 7 levels before cleaning). The row filter
  before the fixed-width column split now restricts to 6-7 levels, so malformed
  deeper entries are excluded instead of crashing the whole filter.

### Changed — breaking

- `make_classifier()`, `classify_sequences()`, `conditional_prob.seq_to_kmers_database()`,
  `batch_classifier.ClassifyAll.calc_kmer_mat()`/`.classify()`, and
  `training_data.filter_train_set()` now take explicit, named parameters instead
  of an untyped `**kwargs` catch-all (e.g. `kmer_size`, `min_confidence`,
  `n_levels`, `terms`, `threshold`, and others — see each function's docstring
  for the full list). Two practical effects: a misspelled or unrecognized
  keyword argument now raises `TypeError` immediately instead of being silently
  ignored (this is what let a `kmers_size`/`kmer_size` typo go unnoticed for a
  while), and `make_classifier()`, `classify_sequences()`, and
  `filter_train_set()` are now keyword-only past their first one or two
  positional arguments — any code passing their options positionally will need
  to switch to keyword arguments.

### Removed

- Deleted unused, superseded code with no callers outside their own tests:
  `get_kmer_db.py` (the whole module — `GetKmerDB`, `load_db`),
  `conditional_prob.build_database()`, and `kmers.bootstrap()`/`bootstrap_kmers()`.
  None of these were used by the actual classification pipeline.

### Tests

- Added a regression test for the `filter_train_set` fix using real taxa strings
  pulled from a raw SILVA release, spanning the full 1-7 level range plus a
  constructed 8-level (malformed) case.
- Removed tests tied to the deleted dead code.

## [0.3.4] - 2026-08-29

### Performance

*All benchmarks below were measured on a 2020 MacBook Pro, 2.3 GHz 8-Core Intel Core i9,
16 GB RAM, running macOS 12.7.6.*

- **`make_classifier()` is dramatically faster on large reference databases.** Building a
  classifier from a full SILVA database (~330k reference sequences) dropped from **202s to
  68.5s (2.9x faster)**; a smaller RDP-scale build (~24.6k sequences) dropped from 8.7s to
  6.7s. The bottleneck was never the cython conditional-probability calculation — it was
  k-mer detection accumulating results as Python lists of boxed integers (9x memory overhead
  vs. a compact numpy array), which caused severe OS-level swap thrashing on large databases.
  - Fixed by:
    - Returning numpy arrays instead of Python lists from `detect_kmers`/`detect_kmer_indices`.
    - Streaming multiprocessing results directly into a preallocated output buffer
      (`pool.imap` + upfront size estimate) instead of collecting every per-sequence result
      into memory before compacting them.
    - Peak memory for k-mer detection on the SILVA-scale build fell from >12GB of swap usage
      to under 3GB resident, with no swapping.
- **`classify_sequences()` taxonomy summarization is 11-18x faster.** The per-sequence
  bootstrap-consensus step (`batch_classifier.summarize()`) used a `Counter`-based majority
  vote per taxonomic level per sequence; replaced with a single vectorized pass
  (`bootstrap.bootstrap_consensus_batch`, using `scipy.stats.mode`) across all sequences at
  once. Verified byte-identical output against the previous implementation on real data.
- Replaced `pandarallel` with a single shared, lazily-created `multiprocessing.Pool`
  (`_worker_pool.py`), reused for the life of the process instead of being torn down and
  recreated on every call. Removes a dependency that was being re-initialized redundantly on
  every import (and printing an unsuppressible startup banner) in favor of a pattern already
  used elsewhere in the codebase.

### Fixed

- `scipy` was used directly (`batch_classifier.py`) but was never declared as a project
  dependency — added to `pyproject.toml`.
- `make_classifier()` accepted a `kmers_size` keyword while `classify_sequences()` expected
  `kmer_size` for the same concept; both now consistently use `kmer_size`.

### Removed

- Dropped unused, superseded classification helpers (`kmers.classify`, `kmers.classify_bs`,
  `kmers.classify_bootstraps`) and the `pandarallel` dependency (along with its transitive
  dependencies `dill` and `psutil`).

### Changed

- Renamed `batched_classify.py` to `batch_classifier.py` and removed a leftover demo/debug
  `__main__` block that referenced a hardcoded local file path.

### Tests

- Added `tests/test_batch_classifier.py`, including a test that specifically exercises the
  chunked bootstrap-classification code path (forcing multiple chunk passes to confirm
  results are unaffected by chunk size).
- Removed dead tests tied to the deleted classification helpers; cleaned up stray debug
  `print()` statements and a no-op assertion in `tests/test_kmers.py`.
