# phylotypy

[![PyPI version](https://img.shields.io/pypi/v/phylotypy?cacheBuster=1)](https://pypi.org/project/phylotypy/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Naive Bayesian Classifier for 16S rRNA gene sequences, inspired by the
[phylotypr](https://github.com/riffomonas/phylotypr) R package by Riffomonas. Designed for classifying amplicon sequence variants (ASVs) 
from DADA2, QIIME2, or raw FASTA files against a reference database of 16S 
rRNA sequences. The RDP training data is provided here in the data directory located at 
the github repository. But Silva and others can be used.

Thanks to Riffomonas for the inspiration — check out the videos on his
[YouTube channel](https://youtube.com/playlist?list=PLmNrK_nkqBpIZlWa3yGEc2-wX7An2kpCL&si=LmHDV02K5_wb6C0j).

---

## Performance

The full RDP reference database takes **~7.5 seconds** and
the full Silva reference database (genus level) takes **~19 seconds**
on a 2020 Apple Intel MacBook Pro with 16Gb of RAM.  Newer systems should 
see a substantial increase in performance.
---

## How to Install

Using pip:
```bash
pip install phylotypy
```

Using uv (recommended — [how to install uv](https://docs.astral.sh/uv/getting-started/installation/)):
```bash
uv pip install phylotypy
```

> **Note**: Intel Mac (x86_64) users are limited to numba 0.62.1, 
> which is pinned in this package. Apple Silicon (M-series) users 
> are not affected.

---

## Training Data

Download the RDP reference training set and an example dataset before classifying:

| File | Description |
|------|-------------|
| [rdp_16S_v19.dada2.fasta](https://github.com/csaltikov/phylotypy/blob/dca2326c0f91fff49bc3dc559df5d66fe9ee6953/data/rdp_16S_v19.dada2.fasta) | RDP trainset19072023, DADA2 format |
| [dna_moving_pictures.fasta](https://github.com/csaltikov/phylotypy/blob/dca2326c0f91fff49bc3dc559df5d66fe9ee6953/data/dna_moving_pictures.fasta) | Example dataset (Moving Pictures study) |
The training data fasta descriptions should contain a taxonomy header only. This RDP training data is
genus level, but phylotypy can accept species level training data.
```
"Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"
```
The taxon string in the fasta description should follow the semicolon-separated format like this:
```
>Bacteria;Pseudomonadota;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Citrobacter
TAGAGTTTGATCCATGGCTCAGATTGAACGCTGGCGGCAGGCCTAACAC.....
```

---

## Reference Database: Taxonomic Levels

Every reference sequence needs a taxonomy string in its FASTA header (see the format
above), and **all headers in the file must have the same number of `;`-delimited
levels**, e.g. `Kingdom;Phylum;Class;Order;Family;Genus` is 6 levels. This matters
because `make_classifier()` and `classify_sequences()` compare taxonomy strings
assuming a fixed depth; a reference set with mixed depths will fail, either right away
or partway through classification.

### 1. Headers already share the same depth

If your reference fasta is already uniform (true of the RDP training set used above,
at 6 levels/genus), just load and build normally:

```python
from phylotypy import read_fasta, classifier

rdp = read_fasta.read_taxa_fasta("rdp_16S_v19.dada2.fasta", is_ref=True)
database = classifier.make_classifier(rdp, verbose=True)
```

Pass `is_ref=True` to `read_taxa_fasta()` to check this up front: it counts the
semicolon-delimited levels in every header, and raises a `ValueError` naming the file
and the depths it found if they're not all the same. So a bad reference file is
caught immediately, not partway through building the classifier or later during
classification. Leave `is_ref` at its default (`False`) when reading sequences
you're going to *classify*, since those don't carry a fixed-depth taxonomy string.

### 2. Fixing ragged headers

"Ragged" headers means the taxonomy strings in the reference fasta don't all reach the
same depth. Some sequences are annotated down to Genus, others only down to Class or
Order:

```
>Bacteria;Firmicutes;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus
TAGAGTTTGATCCTGGCTCAG...
>Bacteria;Firmicutes;Bacilli
TAGAGTTTGATCCTGGCTCAG...
```

This is common with broader databases like Silva. Trying to build a classifier from
ragged reference data (or reading it with `is_ref=True`) raises:

```
ValueError: ref_db contains taxonomy strings with inconsistent numbers of taxonomic levels:
id
3      1204
6     88213
All 'id' entries must have the same number of ';'-delimited levels before building a
classifier, or classify_sequences() will fail later on.
Options:
  - Fix/pad the taxonomy strings in ref_db so every entry has the same depth.
  - Call make_classifier(ref_db, filter_db=True) to filter ref_db down to a single
    consistent depth (see training_data.filter_train_set).
  - Pass n_levels=<int> together with filter_db=True to control which depth is kept.
```

Fix it with `training_data.filter_train_set()`, which drops sequences that don't match
a target depth (`n_levels`) and strips out common noise terms (`Eukaryota`,
`metagenome`, `Candidatus`, etc.. See `training_data.DEFAULT_NOISE_TERMS`):

```python
from phylotypy import read_fasta, training_data, classifier

# is_ref left at False here — the file is ragged, so the check would raise
silva = read_fasta.read_taxa_fasta("silva_nr99.fasta.gz")

filtered = training_data.filter_train_set(silva, n_levels=6, verbose=True)
database = classifier.make_classifier(filtered, verbose=True)
```

With `verbose=True`, `filter_train_set()` reports how many sequences it dropped:

```
Removed 12,458 sequences with taxonomic depth != 6 (kept 158,970)
```

Or skip the manual filter step and let `make_classifier()` do it for you:

```python
database = classifier.make_classifier(silva, filter_db=True, verbose=True)
```

This runs `filter_train_set()` internally, auto-detecting `n_levels` from the
majority depth in your data if you don't pass one, and prints the same before/after
counts when `verbose=True`.

---

## Quick Start

### 1. Load training data and sequences to classify
```python
from phylotypy import classifier, results, read_fasta

rdp = read_fasta.read_taxa_fasta("rdp_16S_v19.dada2.fasta")
moving_pics = read_fasta.read_taxa_fasta("dna_moving_pictures.fasta")
```

### 2. Train the classifier
```python
# Accepts fasta or csv/tsv with 'id' and 'sequence' column names
database = classifier.make_classifier(rdp, verbose=True)

# If the reference has ragged (inconsistent) taxonomic levels, use filter_db=True
# to remove the mismatched records — see "Reference Database: Taxonomic Levels" above
database = classifier.make_classifier(rdp, filter_db=True, verbose=True)
```

### 3. Classify sequences
```python
classified = classifier.classify_sequences(moving_pics, database)
```

### 4. Format and export results
```python
classified = results.summarize_predictions(classified)
print(classified.columns)
```

Output:
```
Index(['id', 'sequence', 'classification', 'Kingdom', 'Phylum', 'Class',
       'Order', 'Family', 'Genus', 'observed', 'lineage'],
      dtype='object')
```

```python
classified.to_csv("classified_results.csv")
```

---

## Complete Code Block

```python
from phylotypy import classifier, results, read_fasta

rdp = read_fasta.read_taxa_fasta("rdp_16S_v19.dada2.fasta")
moving_pics = read_fasta.read_taxa_fasta("dna_moving_pictures.fasta")

database = classifier.make_classifier(rdp)

classified = classifier.classify_sequences(moving_pics, database)
classified = results.summarize_predictions(classified)
print(classified.head())

classified.to_csv("classified_results.csv")
```

---

## Example Classification Output

Taxonomic levels (Domain → Genus) are semicolon-separated. Numbers in parentheses
represent bootstrap confidence scores. The default confidence threshold is 80%.

```
Bacteria(100);Pseudomonadota(99);Alphaproteobacteria(99);Rhodospirillales(99);Acetobacteraceae(99);Roseomonas(83)

Bacteria(99);Bacteroidota(97);Bacteroidia(93);Bacteroidales(93);Bacteroidales_unclassified(93);Bacteroidales_unclassified(93)

Bacteria(100);Bacteroidota(100);Bacteroidia(100);Bacteroidales(100);Bacteroidaceae(100);Bacteroides(100)
```

---

## Working with Your Own Data

phylotypy works with FASTA files from DADA2, QIIME2, or any standard pipeline.
See [read_fasta.py](https://github.com/csaltikov/phylotypy/blob/dca2326c0f91fff49bc3dc559df5d66fe9ee6953/src/phylotypy/utilities/read_fasta.py) for utilities to load
and convert sequence data into the required format.

A complete walkthrough is available in [vignette.py](https://github.com/csaltikov/phylotypy/blob/dca2326c0f91fff49bc3dc559df5d66fe9ee6953/vignette.py).

---

## Requirements

Dependencies are installed automatically via pip. See
[pyproject.toml](https://github.com/csaltikov/phylotypy/blob/main/pyproject.toml)
for the full list.

---

## Citation

If you use phylotypy in your research, please cite:

- Wang, Q., Garrity, G.M., Tiedje, J.M., Cole, J.R. (2007) Naive Bayesian Classifier
  for Rapid Assignment of rRNA Sequences into the New Bacterial Taxonomy.
  *Applied and Environmental Microbiology*, 73(16), 5261–5267.
- Schloss PD.2025.phylotypr: an R package for classifying DNA sequences. 
  Microbiol Resour Announc14:e01144-24.https://doi.org/10.1128/mra.01144-24
- Saltikov, C. (2024) phylotypy: Python implementation of a Naive Bayesian 16S rRNA classifier.
  https://github.com/csaltikov/phylotypy
