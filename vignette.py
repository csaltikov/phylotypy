##
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import phylotypy

##
"""
Notes:
Load data, can use trainset.py to download and produce trainset19_072023_db.csv
or do it yourself another way.
"""
db_file_path = Path("data/trainset19_072023_db.csv")
db_test = pd.read_csv(db_file_path)

# db_test = db_test[db_test["Class"].str.contains("Alphaproteobacteria")]

genus_mapper = phylotypy.genera_index_mapper(db_test["Genus"])
# remove the trailing ; of the taxonomy string
db_test = (db_test.assign(factors=lambda df_: pd.factorize(df_["Genus"])[0],
                          taxonomy=lambda df_: df_["taxonomy"].str.rstrip(";")
                          )
           )
print(f"Size of the database: {db_test.shape}")

##
X_ref, y_ref = db_test["sequences"].tolist(), db_test["taxonomy"].tolist()

X_train, X_test, y_train, y_test = train_test_split(X_ref, y_ref, test_size=0.1, random_state=42)
print(f"Testing {len(X_test)} sequences")

##
# Reload the module in case I edit the code
kmersize = 8
classify = phylotypy.Phylotypy()
classify.fit(X_train, y_train, kmer_size=kmersize, verbose=False)
classify.verbose = True
classified_test_seqs = classify.predict(X_test, y_test, kmer_size=kmersize, boot=10)

##
classified_df = phylotypy.summarize_predictions(classified_test_seqs, y_test)
print(classified_df.head())

##
# Check accuracy
accuracy = accuracy_score(y_test, classified_df["full lineage"]) * 100
print(f"accuracy score: {accuracy:.1f}%")
