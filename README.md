# phylotypy
Naive Bayesian Classifier for 16S rRNA gene sequence data

Porting Riffomonas's CodeClub R package, phylotypr to python: https://github.com/riffomonas/phylotypr

It's been a great challenge learning how to interpret the R code into Python with minimal use of extra libraries.

It's best to clone the repository.  Run vigentte.py to see if everything works.

Training the model with the full reference database from RDP takes about 30 seconds on my MacBook Pro.

You can modify the vignette at the end to classify your own sequences. I've done this using DADA2's output files.

There's also [read_fasta.py](src/phylotypy/utilities/read_fasta.py) that lets you take a fasta file of DNA seqences and process them into a dataframe for runing this classifier.

I made a separate vignette, [vignette.py](vignette.py) on how to do this and classify 16S sequence data from QIIME, DADA2, or text files.

Thanks Riffomonas for the inspiration.  Check out the videos on his Youtube channel https://youtube.com/playlist?list=PLmNrK_nkqBpIZlWa3yGEc2-wX7An2kpCL&si=LmHDV02K5_wb6C0j

## How to install
```
pip install git+https://github.com/csaltikov/phylotypy.git

or if using uv (recommended)

uv pip install git+https://github.com/csaltikov/phylotypy.git
```

## How to get started:
First download the training data, RDP's trainset19072023, either from https://mothur.org/wiki/rdp_reference_files/
 You can use the one in the [data](data/) directory called [rdp_16S_v19.dada2.fasta](data/rdp_16S_v19.dada2.fasta)

I processed the latest rdp reference data into a format that will work here and for DADA2.

The taxonomy string looks like this semicolon separated:
```
Bacteria;Phylum;Class;Order;Family;Genus

for example:
>Bacteria;Pseudomonadota;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Citrobacter
TAGAGTTTGATCCATGGCTCAGATTGAACGCTGGCGGCAGGCCTAACAC.....
```

1. Load the training data and sequences to be classified
```
from pathlib import Path
from phylotypy import classifier, results, read_fasta

rdp = read_fasta.read_taxa_fasta("data/rdp_16S_v19.dada2.fasta")
moving_pics = read_fasta.read_taxa_fasta("data/dna_moving_pictures.fasta")
```
2. Create the classifier. We'll call it database
```
database = classifier.make_classifier(rdp)
```
3. Classify the sequences
```
classified = classifier.classify_sequences(moving_pics, database)
```
4. Format the output
```
classified = results.summarize_predictions(classified)
print(classified.columns)
```
Output:
```
>>> Index(['id', 'sequence', 'classification', 'Kingdom', 'Phylum', 'Class',
       'Order', 'Family', 'Genus', 'observed', 'lineage'],
      dtype='object')
```
```
print(classified["classification"].head())
```
Output:
```
   0    Bacteria(100);Bacteroidota(100);Bacteroidia(10...
   1    Bacteria(100);Pseudomonadota(100);Betaproteoba...
   2    Bacteria(100);Bacillota(100);Bacilli(100);Lact...
   3    Bacteria(100);Bacteroidota(100);Bacteroidia(10...
   4    Bacteria(100);Bacteroidota(100);Bacteroidia(10...
   Name: classification, dtype: object
```
Format the results using results.summarize_predictions() function.
The output is a pandas dataframe and can be saved to csv.
```
from phylotypy import results
classified = results.summarize_predictions(classified)
print(classified.head())

classified.to_csv("classified_results.csv")

```
## Example classification output:
The taxonomic levels "Domain", "Phylum", "Class", "Order", "Family", "Genus" are separated by ";".  The numbers in the () represent the confidence in the classificaiton.  The default confidence is 80%.
```
>>> Bacteria(100);Pseudomonadota(99);Alphaproteobacteria(99);Rhodospirillales(99);Acetobacteraceae(99);Roseomonas(83)

>>> Bacteria(99);Bacteroidota(97);Bacteroidia(93);Bacteroidales(93);Bacteroidales_unclassified(93);Bacteroidales_unclassified(93)

```
## Complete code block:
```
from pathlib import Path
from phylotypy import classifier, results, read_fasta

rdp = read_fasta.read_taxa_fasta("data/rdp_16S_v19.dada2.fasta")
moving_pics = read_fasta.read_taxa_fasta("data/dna_moving_pictures.fasta")

database = classifier.make_classifier(rdp)

classified = classifier.classify_sequences(moving_pics, database)
classified = results.summarize_predictions(classified)
print(classified.head())
```
