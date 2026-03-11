# phylotypy

![PyPI version](https://badge.fury.io/py/phylotypy.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A Naive Bayesian Classifier for 16S rRNA gene sequences, inspired by the
[phylotypr](https://github.com/riffomonas/phylotypr) R package by Riffomonas.
Designed for classifying amplicon sequence variants (ASVs) from DADA2, QIIME2,
or raw FASTA files against the RDP reference database.

Thanks to Riffomonas for the inspiration — check out the videos on his
[YouTube channel](https://youtube.com/playlist?list=PLmNrK_nkqBpIZlWa3yGEc2-wX7An2kpCL&si=LmHDV02K5_wb6C0j).

---

## Performance

Training on the full RDP reference database takes **~30 seconds** on a 2020 Apple Intel MacBook Pro.

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

---

## Training Data

Download the RDP reference training set and an example dataset before classifying:

| File | Description |
|------|-------------|
| [rdp_16S_v19.dada2.fasta](https://raw.githubusercontent.com/csaltikov/phylotypy/refs/heads/main/data/rdp_16S_v19.dada2.fasta) | RDP trainset19072023, DADA2 format |
| [dna_moving_pictures.fasta](https://raw.githubusercontent.com/csaltikov/phylotypy/refs/heads/main/data/dna_moving_pictures.fasta) | Example dataset (Moving Pictures study) |

The RDP training data uses semicolon-separated taxonomy strings in this format:
```
>Bacteria;Pseudomonadota;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Citrobacter
TAGAGTTTGATCCATGGCTCAGATTGAACGCTGGCGGCAGGCCTAACAC.....
```

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
database = classifier.make_classifier(rdp)
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
See [read_fasta.py](src/phylotypy/utilities/read_fasta.py) for utilities to load
and convert sequence data into the required format.

A complete walkthrough is available in [vignette.py](vignette.py).

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
- Saltikov, C. (2024) phylotypy: Python implementation of a Naive Bayesian 16S rRNA classifier.
  https://github.com/csaltikov/phylotypy
