##
from pathlib import Path
import pickle
import time

import pandas as pd

try:
    import phylotypy
except ImportError:
    import phylotypy.phylotypy as phylotypy


##
db_file_path = Path("../data/trainset19_072023_db.csv")
db = pd.read_csv(db_file_path)

# remove the trailing ; of the taxonomy string
db = (db.assign(taxonomy=lambda df_: df_["taxonomy"].str.rstrip(";")))
print(f"Size of the database: {db.shape}")

X_ref, y_ref = db["sequences"].tolist(), db["taxonomy"].tolist()

##
# Trains the model
kmersize = 8
classify = phylotypy.Phylotypy()

start = time.time()

classify.fit(X_ref, y_ref, kmer_size=kmersize, verbose=True)
classify.verbose = True

end = time.time()

print(f"Run time {(end - start):.1f} seconds")

##
# Save model for later use
with open('classifier.pkl', 'wb') as f:
    pickle.dump(classify, f)
