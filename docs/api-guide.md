# Using the API

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
# to remove the mismatched records — see docs/training-data.md
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
```python
Index(['id', 'sequence', 'classification', 'Kingdom', 'Phylum', 'Class',
       'Order', 'Family', 'Genus', 'observed', 'lineage'],
      dtype='object')
```

```python
classified.to_csv("classified_results.csv")
```

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

## Example Classification Output

Taxonomic levels (Domain → Genus) are semicolon-separated. Numbers in parentheses
represent bootstrap confidence scores. The default confidence threshold is 80%.

```python
Bacteria(100);Pseudomonadota(99);Alphaproteobacteria(99);Rhodospirillales(99);Acetobacteraceae(99);Roseomonas(83)

Bacteria(99);Bacteroidota(97);Bacteroidia(93);Bacteroidales(93);Bacteroidales_unclassified(93);Bacteroidales_unclassified(93)

Bacteria(100);Bacteroidota(100);Bacteroidia(100);Bacteroidales(100);Bacteroidaceae(100);Bacteroides(100)
```
