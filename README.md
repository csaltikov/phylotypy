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

## Here's an example on how to start:
1. First download the training data, RDP's trainset19072023, either from https://mothur.org/wiki/rdp_reference_files/ or use the code below. This will create a directory called 'data' where the training data will be downloaded and processed into a csv file for importing into Pandas.
```
from training_data import rdp_train_set_19

rdp_train_set_19()
```

2. Load the training data
```
import pandas as pd

db_file_path = "data/trainset19_072023_db.csv"
db = pd.read_csv(db_file_path)

# the RDP train set has a trailing ';' let's remove it
db["taxonomy"] = db["taxonomy"].str.rstrip(";")
```
3. Create the training data for the classifer
```
X_ref, y_ref = db["sequences"].tolist(), db["taxonomy"].tolist()
```
4. Train the classifer
```
import phylotypy

classify = phylotypy.Phylotypy()
classify.fit(X_ref, y_ref, verbose=True)
```
5. Classify some 16S rRNA gene sequences.  Here we will use the example from QIIME2, Moving Pictures data, https://docs.qiime2.org/2024.2/tutorials/moving-pictures/.  The data came from Classifying QIIME2 Moving Pictures rep-seqs-dada2.qza
```
from utilities import fasta_to_dataframe

moving_pic = fasta_to_dataframe("data/dna_moving_pictures.fasta")


X_mov_pic = moving_pic["Sequence"].to_list()  # DNA sequences as a list
y_mov_pic = moving_pic["SequenceName"].to_list()  # Sequence names as a list


predict_mov_pic = classify.predict(X_mov_pic, y_mov_pic)  # train the classifier

predict_mov_pic_df = phylotypy.summarize_predictions(predict_mov_pic)  # results are a Pandas dataframe
print(predict_mov_pic_df[["id", "classification"]])  # the full classifcation is in the 'classification' column
```
## Example classification output:
The taxonomic levels "Domain", "Phylum", "Class", "Order", "Family", "Genus" are separated by ";".  The numbers in the () represent the confidence in the classificaiton.  The default confidence is 80%.
```
>>> Bacteria(100);Pseudomonadota(99);Alphaproteobacteria(99);Rhodospirillales(99);Acetobacteraceae(99);Roseomonas(83),Bacteria(100),Pseudomonadota(99),Alphaproteobacteria(99),Rhodospirillales(99),Acetobacteraceae(99),Roseomonas(83)

>>> Bacteria(99);Bacteroidota(97);Bacteroidia(93);Bacteroidales(93);Bacteroidales_unclassified(93);Bacteroidales_unclassified(93),Bacteria(99),Bacteroidota(97),Bacteroidia(93),Bacteroidales(93),Bacteroidales_unclassified(93),Bacteroidales_unclassified(93)

```

6. Now you can keep using classify.predict(X, y) with your own data.  Make sure the sequences and names are lists
```
my_data = "path/to/my/sequences" # change path to your fasta sequence file
my_data_df = fasta_to_dataframe(my_data)

X = my_data_df["Sequence"].to_list()
y = my_data_df["SequenceName"].to_list()

predict = classify.predict(X, y)

predict_df = phylotypy.summarize_predictions(predict)
print(predict_df[["id", "classification"]])
```

## Complete code block:
```
import pandas as pd

import phylotypy
from training_data import rdp_train_set_19
from utilities import fasta_to_dataframe

# dowload/format training data
rdp_train_set_19()

db_file_path = "data/trainset19_072023_db.csv"
db = pd.read_csv(db_file_path)

# remove the trailing ; of the taxonomy string
db["taxonomy"] = db["taxonomy"].str.rstrip(";")

X_ref, y_ref = db["sequences"].tolist(), db["taxonomy"].tolist()

classify = phylotypy.Phylotypy()
classify.fit(X_ref, y_ref, verbose=True)  # train the model

moving_pic = fasta_to_dataframe("data/dna_moving_pictures.fasta")

X_mov_pic = moving_pic["Sequence"].to_list()  # DNA sequences as a list
y_mov_pic = moving_pic["SequenceName"].to_list()  # Sequence names as a list

predict_mov_pic = classify.predict(X_mov_pic, y_mov_pic)  # train the classifier

predict_mov_pic_df = phylotypy.summarize_predictions(predict_mov_pic)  # results are a Pandas dataframe
print(predict_mov_pic_df[["id", "classification"]])  # the full classifcation is in the 'classification' column
```
 
