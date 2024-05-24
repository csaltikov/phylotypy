# phylotypy
Naive Bayesian Classifier for 16S rRNA gene sequence data

Porting Riffomonas's CodeClub R package, phylotypr to python: https://github.com/riffomonas/phylotypr

It's been a great challenge learning how to interpret the R code into Python with minimal use of extra libraries.

It's best to clone the repository.  Run vigentte.py to see if everything works.

If it does then you can classify your own sequences

```
# prepare your data:
import utilities  # you need to install Biopython

unknown_seq = utilities.fasta_to_dataframe("data/orio_16s.txt") # pd.Dataframe
print(unknown_seq.columns)
```
```
>>> Index(['SequenceName', 'Sequence'], dtype='object')
```
Convert your data into two separate lists, one for sequence data and the other of the names/ids of each sequence
```
X_unknown = unknown_seq["Sequence"].to_list()
y_unknown = unknown_seq["SequenceName"].to_list()
```

Now, you need to train the model, first you need to get the training data
```
from pathlib import Path
import pandas as pd
import phylotypy


db_file_path = Path("data/trainset19_072023_db.csv")
db_test = pd.read_csv(db_file_path)

# remove the trailing ; of the taxonomy string
db_test = (db_test.assign(taxonomy=lambda df_: df_["taxonomy"].str.rstrip(";")))
print(f"Size of the database: {db_test.shape}")
```
```
>>> Size of the database: (24642, 10)
```
```
X_ref = db_test["sequences"].tolist(),
y_ref = db_test["taxonomy"].tolist()
```
Now we are ready to train the model predict classifications
```
kmersize = 8
classify = phylotypy.Phylotypy()
classify.fit(X_ref, y_ref, kmer_size=kmersize, verbose=True)
```
Now you are ready to classify your sequence(s), and feed the predictions into the summarize_predictions()
to see Pandas dataframe of the results
```
classify_unknown = classify.predict(X_unknown, y_unknown)

classify_unknown_df = phylotypy.summarize_predictions(classify_unknown, y_unknown)
print(classify_unknown_df)
```
