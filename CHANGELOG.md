# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
