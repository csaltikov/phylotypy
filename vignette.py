##
from pathlib import Path
import time

import pandas as pd

from utilities import fasta_to_dataframe
import phylotypy


##
"""
Notes:
Load data, can use training_data.py to download and produce trainset19_072023_db.csv
or do it yourself another way.
"""

db_file_path = Path("data/trainset19_072023_db.csv")
db_test = pd.read_csv(db_file_path)

# remove the trailing ; of the taxonomy string
db_test = (db_test.assign(taxonomy=lambda df_: df_["taxonomy"].str.rstrip(";")
                          )
           )
print(f"Size of the database: {db_test.shape}")

##
X_ref, y_ref = db_test["sequences"].tolist(), db_test["taxonomy"].tolist()

##
# Reload the module in case I edit the code
kmer_size = 8
classify = phylotypy.Phylotypy()
start = time.time()
classify.fit(X_ref, y_ref, kmer_size=kmer_size, verbose=True)
classify.verbose = False
end = time.time()
print(f"Run time {(end - start):.1f} seconds")

##
# Classifying QIIME2 Moving Pictures rep-seqs-dada2.qza
# https://docs.qiime2.org/2024.2/tutorials/moving-pictures/
moving_pic = fasta_to_dataframe("data/dna_moving_pictures.fasta")

# prepare the sequences and sequence name lists
X_mov_pic = moving_pic["Sequence"].to_list()
y_mov_pic = moving_pic["SequenceName"].to_list()

# Classify
predict_mov_pic = classify.predict(X_mov_pic, y_mov_pic)

# Put results in a dataframe
predict_mov_pic_df = phylotypy.summarize_predictions(predict_mov_pic)
print(predict_mov_pic_df[["id", "Genus"]])

##
# Testing a single organism sequences
orio = fasta_to_dataframe("data/orio_16s.txt")  # pd.Dataframe

# prepare the sequences and sequence name lists
X_unk = orio["Sequence"].to_list()
y_unk = orio["SequenceName"].to_list()

# Classify the sequence
predict_orio = classify.predict(X_unk, y_unk)

# Report the results in a dataframe
predict_orio_df = phylotypy.summarize_predictions(predict_orio)
print(predict_orio_df[["id", "Genus"]])
