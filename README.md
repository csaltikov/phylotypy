# phylotypy
Naive Bayesian Classifier for 16S rRNA gene sequence data

Porting Riffomonas's CodeClub R package, phylotypr to python: https://github.com/riffomonas/phylotypr

It's been a great challenge learning how to interpret the R code into Python with minimal use of extra libraries.

It's best to clone the repository.  Run vigentte.py to see if everything works.

Training the model with the full reference database from RDP takes about 30 seconds on my MacBook Pro.

You can modify the vignette at the end to classify your own sequences. I've done this using DADA2's output files.

There's also utility.py that lets you take a fasta file of DNA seqences and process them into a dataframe for runing this classifier.

I'll make a separate vignette on how to do this and classify 16S sequence data from QIIME, DADA2, or text files.

Thanks Riffomonas for the inspiration.  Check out the videos on his Youtube channel https://youtube.com/playlist?list=PLmNrK_nkqBpIZlWa3yGEc2-wX7An2kpCL&si=LmHDV02K5_wb6C0j

## How to install
```
pip install git+https://github.com/csaltikov/phylotypy.git

or if using uv (recommended)

uv pip install git+https://github.com/csaltikov/phylotypy.git
```

## How to get started:
First download the training data, RDP's trainset19072023, either from https://mothur.org/wiki/rdp_reference_files/

I processed the latest rdp reference data into a csv file.

The taxonomy string looks like:
```
Bacteria;Phylum;Class;Order;Family;Genus
```

1. Load the training data
```
import pandas as pd

db_file_path = "data/trainset19_072023_db.csv"
db = pd.read_csv(db_file_path)
```
2. Create the training data for the classifer
```
X_ref, y_ref = db["sequence"], db["id"]
```
3. Train the classifer
```
from phylotypy import predict

classify = predict.Classify()
classify.multi_processing = True
classify.fit(X_ref, y_ref, multi=True, n_cpu=12)
```
4. Classify some 16S rRNA gene sequences.  Here we will use the example from QIIME2, Moving Pictures data, https://docs.qiime2.org/2024.2/tutorials/moving-pictures/.  The data came from Classifying QIIME2 Moving Pictures rep-seqs-dada2.qza
```
from utilities import fasta_to_dataframe

moving_pic = read_fasta.read_taxa_fasta("data/dna_moving_pictures.fasta")

X_mov_pic = moving_pic["Sequence"]  # DNA sequences as a list
y_mov_pic = moving_pic["SequenceName"]  # Sequence names as a list

predict_mov_pic = classify.predict(X_mov_pic, y_mov_pic)  # train the classifier

predict_mov_pic_df = predict.summarize_predictions(predict_mov_pic)  # results are a Pandas dataframe
print(predict_mov_pic_df[["id", "classification"]])  # the full classifcation is in the 'classification' column
```
## Example classification output:
The taxonomic levels "Domain", "Phylum", "Class", "Order", "Family", "Genus" are separated by ";".  The numbers in the () represent the confidence in the classificaiton.  The default confidence is 80%.
```
>>> Bacteria(100);Pseudomonadota(99);Alphaproteobacteria(99);Rhodospirillales(99);Acetobacteraceae(99);Roseomonas(83)

>>> Bacteria(99);Bacteroidota(97);Bacteroidia(93);Bacteroidales(93);Bacteroidales_unclassified(93);Bacteroidales_unclassified(93)

```

5. Now you can keep using classify.predict(X, y) with your own data.  Make sure the sequences and names are lists
```
my_data = "path/to/my/sequences" # change path to your fasta sequence file
my_data_df = fasta_to_dataframe(my_data)

X = my_data_df["sequence"]
y = my_data_df["id"]

predict = classify.predict(X, y)

predict_df = phylotypy.summarize_predictions(predict)
print(predict_df[["id", "classification"]])
```

## Complete code block:
```
import pandas as pd

from phylotypy import predict
from phylotypy.utilities import utilities

db_file_path = "data/trainset19_072023_db.csv"
db = pd.read_csv(db_file_path)

X_ref, y_ref = db["sequence"], db["id"]

classify = predict.Classify()
classify.multi_processing = True
classify.fit(X_ref, y_ref, multi=True, n_cpu=12)  # train the model

moving_pic = read_fasta.read_taxa_fasta("data/dna_moving_pictures.fasta")

X_mov_pic = moving_pic["sequence"]  # DNA sequences as a list
y_mov_pic = moving_pic["id"]  # Sequence names as a list

predict_mov_pic = classify.predict(X_mov_pic, y_mov_pic)  # train the classifier

predict_mov_pic_df = predict.summarize_predictions(predict_mov_pic)  # results are a Pandas dataframe
print(predict_mov_pic_df[["id", "classification"]])  # the full classifcation is in the 'classification' column
```
