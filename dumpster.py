##

from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd

import kmers
import phylotypy


##
def kmer_list():
    fg_data_fil = Path("/Users/caltikov/DataspellProjects/fermegrow/taxa_silva.csv")
    fg_df = pd.read_csv(fg_data_fil, index_col=0).reset_index(names="sequences")
    fg_df.loc[:, "ASV"] = "ASV" + (fg_df.index + 1).astype(str)
    fg_df.set_index("ASV", inplace=True)

    kmer_big_list = []
    for seq in fg_df["sequences"].tolist():
        kmer_list = kmers.seq_to_base4(seq)
        kmer_big_list.append(kmer_list)

    return kmer_big_list


##
db = defaultdict(list)
db["genera"] = np.array(["A;a;A", "A;a;B", "A;a;C", "A;a;A", "A;a;B", "A;b;C"])

y_test = np.array(['Bacteria;Pseudomonadota;Alphaproteobacteria;Hyphomicrobiales;Methylocystaceae;Methylosinus',
 'Bacteria;Pseudomonadota;Alphaproteobacteria;Caulobacterales;Caulobacteraceae;Brevundimonas',
 'Bacteria;Pseudomonadota;Alphaproteobacteria;Hyphomicrobiales;Devosiaceae;Devosia',
 'Bacteria;Pseudomonadota;Alphaproteobacteria;Rhodobacterales;Roseobacteraceae;Sulfitobacter',
 'Bacteria;Pseudomonadota;Alphaproteobacteria;Hyphomicrobiales;Amorphaceae;Acuticoccus',
 'Bacteria;Pseudomonadota;Alphaproteobacteria;Rhodobacterales;Roseobacteraceae;Rubellimicrobium',
 'Bacteria;Pseudomonadota;Alphaproteobacteria;Hyphomicrobiales;Devosiaceae;Devosia',
 'Bacteria;Pseudomonadota;Alphaproteobacteria;Hyphomicrobiales;Devosiaceae;Devosia',
 'Bacteria;Pseudomonadota;Alphaproteobacteria;Sphingomonadales;Erythrobacteraceae;Pontixanthobacter',
 'Bacteria;Pseudomonadota;Alphaproteobacteria;Rhodobacterales;Paracoccaceae;Thioclava'])

taxa_arr =  np.array([['Bacteria', 'Bacteria;Pseudomonadota',
  'Bacteria;Pseudomonadota;Alphaproteobacteria',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae;Aquisalinus'],
 ['Bacteria', 'Bacteria;Pseudomonadota',
  'Bacteria;Pseudomonadota;Alphaproteobacteria',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae;Aquisalinus'],
 ['Bacteria', 'Bacteria;Pseudomonadota',
  'Bacteria;Pseudomonadota;Alphaproteobacteria',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae;Aquisalinus'],
 ['Bacteria', 'Bacteria;Pseudomonadota',
  'Bacteria;Pseudomonadota;Alphaproteobacteria',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae;Aquisalinus'],
 ['Bacteria', 'Bacteria;Pseudomonadota',
  'Bacteria;Pseudomonadota;Alphaproteobacteria',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae;Aquisalinus'],
 ['Bacteria', 'Bacteria;Pseudomonadota',
  'Bacteria;Pseudomonadota;Alphaproteobacteria',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae;Aquisalinus'],
 ['Bacteria', 'Bacteria;Pseudomonadota',
  'Bacteria;Pseudomonadota;Alphaproteobacteria',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae;Aquisalinus'],
 ['Bacteria', 'Bacteria;Pseudomonadota',
  'Bacteria;Pseudomonadota;Alphaproteobacteria',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae;Aquisalinus'],
 ['Bacteria', 'Bacteria;Pseudomonadota',
  'Bacteria;Pseudomonadota;Alphaproteobacteria',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae;Aquisalinus'],
 ['Bacteria', 'Bacteria;Pseudomonadota',
  'Bacteria;Pseudomonadota;Alphaproteobacteria',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae',
  'Bacteria;Pseudomonadota;Alphaproteobacteria;Parvularculales;Parvularculaceae;Aquisalinus']]
                     )

##
def get_max(col):
    """Helper for create_con determines id and fraction"""
    freq = np.apply_along_axis(Counter, axis=0, arr=col).flatten()
    print(freq.shape)
    id_arr = []
    frac_arr = []
    for i in range(freq.shape[0]):
        id, frac = freq[i].most_common(1)[0]
        id_arr.append(id)
        frac_arr.append((frac / len(col) * 100))
    # return id, (frac / len(col) * 100)
    return id_arr, frac_arr


##
def split_class(arr):
    taxonomy_split = [line.split(";") for line in arr]
    return np.array(taxonomy_split)


def create_con(arr: np.array):
    taxa_arr = np.empty(arr.shape, dtype=object)
    taxa = np.empty(arr.shape[1], dtype=object)
    scores =  np.empty(arr.shape[1], dtype=object)
    for i, col in enumerate(arr):
        for j in range(len(col)):
            taxa_arr[i, j] = ";".join(col[:j+1])
    for k in range(taxa_arr.shape[1]):
        taxa[k], scores[k] = get_max(taxa_arr[:, k])
    best_id = filter_scores(scores)
    return taxa[best_id], scores[best_id]


def get_max(col):
    """Helper for create_con determines id and fraction"""
    freq = Counter(col)
    id, frac = freq.most_common(1)[0]
    return id, int(frac / len(col) * 100)


def filter_scores(scores: np.array):
    """Helper for create_con determines finds the best id"""
    threshold = scores >= 80
    scores_filt = np.where(threshold)[0]
    if scores_filt.size > 0:
        return scores_filt[-1]
    else:
        return None


split_arr = split_class(y_test)
print(create_con(split_arr))
