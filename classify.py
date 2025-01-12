__version__ = '0.1.0'
import argparse
import json
from functools import partial
import logging
import multiprocessing as mp
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


try:
    import phylotypy.kmers as kmers
    from phylotypy.utilities import read_fasta
except ImportError:
    import kmers
    from utilities import read_fasta


# Argparse setup
parser = argparse.ArgumentParser(
    description='Classify sequences using a preformatted naïve Bayes classifier.',
    epilog='''Example:
        python classify.py -m models/silva/model_config.json -f data/fg_sequences.fasta -o results/fg_classified.csv

        Output: 
                 id                                     classification
        0  ASV1  d__Bacteria(100);p__Firmicutes(100);c__Clostri...
        1  ASV2  d__Bacteria(100);p__Firmicutes(100);c__Clostri...
        2  ASV3  d__Bacteria(100);p__Bacteroidota(100);c__Bacte...
        3  ASV4  d__Bacteria(100);p__Firmicutes(100);c__Clostri...
        4  ASV5  d__Bacteria(100);p__Actinobacteriota(100);c__A...''',
    formatter_class=argparse.RawDescriptionHelpFormatter
                                 )

parser.add_argument('-m', '--model', required=True, help='Path to the reference database config json file')
parser.add_argument('-f', '--fasta', required=True, help='Path to the input FASTA file')
parser.add_argument('-o', '--output', required=True, help='Path to the output CSV file')
args = parser.parse_args()

##
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)


def db_files(mod_file, genera_file, mod_shape):
    db_ = kmers.KmerDB(conditional_prob=np.memmap(mod_file,
                                                  dtype=np.float32,
                                                  mode="c",
                                                  shape=mod_shape),
                       genera_idx=kmers.genera_str_to_index(np.load(genera_file, allow_pickle=True)),
                       genera_names=np.load(genera_file, allow_pickle=True)
                       )
    # db_ = {"genera": np.load(genera_file, allow_pickle=True),
    #        "conditional_prob": np.memmap(mod_file,
    #                                      dtype=np.float32,
    #                                      mode="c",
    #                                      shape=mod_shape)
    #        }
    return db_


def load_db(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract the necessary information from the config
    model_dir = Path(config['model_dir'])
    model_file = model_dir.joinpath(config['model'])
    genera = model_dir.joinpath(config['genera'])
    model_shape = tuple(config['model_shape'])

    return db_files(model_file, genera, model_shape)


def pipeline(bs_kmer, min_confid: int = 80, n_levels: int = 6):
    """
    Classifies a sequence based on the indices of kmers.

    Parameters:
    -----------
    bs_kmer : list
        kmer indices for a sample, e.g., [3590, 14361, 57447, 33182, ...]
    min_confid : int
        minimum number of kmers to classify 80% acceptable

    Returns:
    --------
    str
        Formatted string of the taxonomic classification, e.g.,
        'Bacteria(100);Pseudomonadota(100);Alphaproteobacteria(100)...'

    Examples:
    ---------
    '>>> bs_kmer = [3590, 14361, 57447, 33182]
    '>>> pipeline(bs_kmer)'
    'Bacteria(100);Pseudomonadota(100);Alphaproteobacteria(100)...'
    """
    classified_list = kmers.classify_bootstraps(bs_kmer, db.conditional_prob)
    classification = kmers.consensus_bs_class(classified_list, db.genera_names)
    classification_filtered = kmers.filter_taxonomy(classification, min_confid)
    return kmers.print_taxonomy(classification_filtered, n_levels)


def pool_bootstrap(kmer_list: list):
    logging.info(f"Starting kmers.bootstrap")
    max_proc = 4
    pool = mp.Pool(processes=max_proc)
    bootstrap_func = partial(kmers.bootstrap, n_bootstraps=100)

    try:
        bs_results_collected = pool.map(bootstrap_func, kmer_list)
        bs_results = [np.array(result) for result in bs_results_collected]
    finally:
        pool.close()
        pool.join()
        logging.info("Finished kmer bootstraps")
    return bs_results


def pool_classify_async(bs_kmer_list, min_confid: int = 80, n_levels: int = 6):
    logging.info("Starting classify/consensus pipeline")
    max_proc: int = 4  # seems to be the optimal
    pool = mp.Pool(processes=max_proc)
    results = []
    collected_bs_results = []
    indices_with_tasks = []
    try:
        # Submit tasks to the pool and keep track of the indices
        for i, indices in enumerate(bs_kmer_list):
            r = pool.apply_async(pipeline, [indices, min_confid, n_levels])
            results.append(r)
            indices_with_tasks.append((i, indices))
        try:
            for j, (r, (index, indices)) in enumerate(zip(results, indices_with_tasks)):
                collected_bs_results.append(r.get())
        except IndexError:
            logging.error(f"IndexError occurred for item at index {index} with indices {indices}, setting to 'unclassified_taxa'")
        except Exception as exp:
            logging.error(f"An error occurred for item at index {index} with indices {indices}: {exp}")
            collected_bs_results.append("unclassified(0)")
    except Exception as e:
        print(e)
    finally:
        pool.close()
        pool.join()
        logging.info("Finished classifying sequences")
    return collected_bs_results


def pool_classify(bs_kmer_list, min_confid: int = 80, n_levels: int = 6):
    logging.info("Starting classify/consensus pipeline")
    max_proc: int = 6  # seems to be the optimal
    pool = mp.Pool(processes=max_proc)

    try:
        args_list = [(kmer_indices, min_confid, n_levels) for kmer_indices in bs_kmer_list]
        collected_bs_results = pool.starmap(pipeline, args_list)
        bs_results = [np.array(results) for results in collected_bs_results]
    finally:
        pool.close()
        pool.join()
        logging.info("Finished pool_classify bootstraps")
    print(len(bs_results))
    return bs_results


def predict(sequences):
    kmer_detect_list = kmers.detect_kmers_across_sequences(sequences, verbose=True)
    bootstrap_kmers = pool_bootstrap(kmer_detect_list)
    results = pool_classify_async(bootstrap_kmers)
    return results


config_file = args.model

if not Path(config_file).exists():
    logging.error(f"Config file {config_file} does not exist")
    sys.exit(1)

db = load_db(config_file)
taxa_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']

##
if __name__ == "__main__":
    fasta_sequences = read_fasta.read_fasta_file(args.fasta)
    output = args.output
    out_path = Path(output)
    if not out_path.parent.exists():
        print(f"Output directory {out_path} does not exist")
        sys.exit(1)

    logging.info(f"Size of the database: {fasta_sequences.shape}")
    X_test, y_test = fasta_sequences["sequence"].tolist(), fasta_sequences["id"].tolist()

    ##
    start = perf_counter()

    res = predict(X_test)

    end = perf_counter()
    print(f"Time taken: {(end - start):.2f}")
    res_df = pd.DataFrame({"id": fasta_sequences["id"], "classification": res})
    print(res_df.head())
    print(res_df.tail())
    print(res_df.shape)

    res_df[taxa_levels] = res_df["classification"].str.split(";", expand=True)
    res_df[taxa_levels] = res_df[taxa_levels].replace(r"\(\d+\)", "", regex=True)

    res_df.to_csv(out_path, index=False)
    logging.info("Done")

