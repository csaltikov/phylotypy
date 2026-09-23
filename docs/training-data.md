# Training Data & Reference Databases

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
