# phylotypy
Naive Bayesian Classifier for 16S rRNA gene sequence data

Porting Riffomonas's CodeClub R package, phylotypr to python: https://github.com/riffomonas/phylotypr

It's been a great challenge learning how to interpret the R code into Python with minimal use of extra libraries.

It's best to clone the repository.  Run vigentte.py to see if everything works.

If it does then you can classify your own sequences

```
# prepare your data:
import utilities
unknown_seq = utilities.fasta_to_dataframe("data/orio_16s.txt") # pd.Dataframe
print(unknown_seq.columns)
>>> Index(['SequenceName', 'Sequence'], dtype='object')
```
Convert your data into two separate lists, one for sequence data and the other of the names/ids of each sequence
```
X_unknown = unknown_seq["Sequence"].to_list()
y_unknown = unknown_seq["SequenceName"].to_list()
```

Now, you need to train the model and then use the predict() function
```
kmersize = 8
classify = phylotypy.Phylotypy()
classify.fit(X_train, y_train, kmer_size=kmersize, verbose=True)

classified_test_seqs = classify.predict(X_test, y_test, kmer_size=kmersize, boot=10)
```
Now you are ready to classify your sequence(s), and feed the predictions into the summarize_predictions()
to see Pandas dataframe of the results
```
classify_unknown = classify.predict(X_unknown, y_unknown)

classify_unknown_df = phylotypy.summarize_predictions(classify_unknown, y_unknown)
print(classify_unknown_df)
```
