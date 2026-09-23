# phylotypy

[![PyPI version](https://img.shields.io/pypi/v/phylotypy?cacheBuster=1)](https://pypi.org/project/phylotypy/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Naive Bayesian Classifier for 16S rRNA gene sequences, inspired by the
[phylotypr](https://github.com/riffomonas/phylotypr) R package by Riffomonas. Designed for classifying amplicon 
sequence variants (ASVs) from DADA2, QIIME2, or raw FASTA files against a reference database of 16S 
rRNA sequences. PhylotyPY was built to run on a laptop with modest hardware. The
project is memory opitmized and takes advantage of a computer's mutiple cpus.

DADA2's assignTaxonomy has no option to save and reuse a classifier. And large reference
fasta files like Silva can tie up a computer for an extended time period. QIIME2 requires conda 
installation and produces artifact files needing to be inter-converted. 

Phylotypy was created to be a drop-in replacement for DADA2 and QIIME2's classifcation steps. 
Phylotypy takes fasta files and csv/tsv file as input options. The output is a tsv
file with columns containing several lineage formats and separate taxonomic levels:

```commandline
# lineage with percent confidence scrores
Bacteria(100);Pseudomonadota(95);Deltaproteobacteria(92);Desulfovibrionales(92);Desulfovibrionaceae(90);Desulfovibrio(80)

# semicolon separate lineage
Bacteria;Pseudomonadota;Deltaproteobacteria;Desulfovibrionales;Desulfovibrionaceae;Desulfovibrio

# qiime formated lineage
k__Bacteria;p__Pseudomonadota;c__Deltaproteobacteria;o__Desulfovibrionales;f__Desulfovibrionaceae;g__Desulfovibrio
```
|Kingdom|Phylum|Class|Order|Family|Genus|
|-------|-------|-----|-----|------|-----|
|Bacteria|Pseudomonadota|Deltaproteobacteria|Desulfovibrionales|Desulfovibrionaceae|Desulfovibrio|

Phylotypy was written from the ground up but using methods presented in
Riffamonas's CodeClub series. I want to thank P. Schloss and Riffomonas for the inspiration to write phylotypy. Check out the videos on his
[YouTube channel](https://youtube.com/playlist?list=PLmNrK_nkqBpIZlWa3yGEc2-wX7An2kpCL&si=LmHDV02K5_wb6C0j).

---
## Performance

The full RDP reference database takes **~7.5 seconds** and
the full Silva reference database (genus level) takes **~19 seconds**
on a 2020 Apple Intel MacBook Pro with 16Gb of RAM. Newer systems should 
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
## Quickstart

Download the RDP reference training set and an example dataset (see
[Training Data](https://github.com/csaltikov/phylotypy/blob/main/docs/training-data.md) for details and download links), then classify
from the command line:

```shell
phylotypy classify --input dna_moving_pictures.fasta \
                   --db rdp_16S_v19.dada2.fasta \
                   --out classified_seqs.tsv \
                   --save-db rdp_classifer.pickle \
                   --terminal-report \
                   --verbose
```

`--save-db` pickles the built classifier so later runs against the same reference
skip rebuilding it. `--terminal-report` prints a bar-chart summary of the results
straight to the terminal:

```text
Phylum-level summary  770 sequences · 20 taxa (19 named)
────────────────────────────────────────────────────────────────────────────────
Bacillota              ████████████████████████████████████████████  277  36.0%
Pseudomonadota         ███████████████████████▉                      150  19.5%
Bacteroidota           █████████████████████▊                        137  17.8%
Bacteria_unclassified  ██████████▋                                    67   8.7%
Actinomycetota         ██████████                                     63   8.2%
...
────────────────────────────────────────────────────────────────────────────────
resolved at phylum: 703/770 (91.3%) · mean confidence 98.4
```

You can also regenerate that same chart later from a results file, without
re-classifying, using `phylotypy report`:

```shell
phylotypy report --input classified_seqs.tsv
```

See [docs/cli-reference.md](https://github.com/csaltikov/phylotypy/blob/main/docs/cli-reference.md) for reusing a saved database,
the terminal report's rank/top options, the `report` subcommand, and full
`--help` output for `build`, `classify`, and `report`.

### Using the API instead

The same steps from a Python script or notebook:

```python
from phylotypy import classifier, results, read_fasta

rdp = read_fasta.read_taxa_fasta("rdp_16S_v19.dada2.fasta")
moving_pics = read_fasta.read_taxa_fasta("dna_moving_pictures.fasta")

database = classifier.make_classifier(rdp)
classified = classifier.classify_sequences(moving_pics, database)
classified = results.summarize_predictions(classified)

classified.to_csv("classified_results.csv")
```

See [docs/api-guide.md](https://github.com/csaltikov/phylotypy/blob/main/docs/api-guide.md) for the full walkthrough, including
formatting/export options and example output.

---
## Documentation

- [docs/training-data.md](https://github.com/csaltikov/phylotypy/blob/main/docs/training-data.md) — downloading reference data, required
  FASTA header format, and fixing "ragged" (inconsistent-depth) taxonomy strings
- [docs/cli-reference.md](https://github.com/csaltikov/phylotypy/blob/main/docs/cli-reference.md) — full CLI usage: building/reusing a
  database, the terminal report, and `--help` output for every subcommand
- [docs/api-guide.md](https://github.com/csaltikov/phylotypy/blob/main/docs/api-guide.md) — step-by-step API usage and a complete code
  example
- [benchmarks/](https://github.com/csaltikov/phylotypy/blob/main/benchmarks/README.md) — speed comparisons against other 16S
  classification tools

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

---
## AI Assistance

Portions of this project (code and documentation) were developed with the assistance of Anthropic's Claude.
