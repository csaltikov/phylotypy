from pathlib import Path
from phylotypy import classifier, results, kmers
from phylotypy import read_fasta


if __name__ == "__main__":
    from time import perf_counter

    db_fasta = Path("data/rdp_16S_v19.dada2.fasta")
    print("Loading test sequences...")
    test_seqs = read_fasta.read_taxa_fasta("data/dna_moving_pictures.fasta")

    print("Loading database sequences...")
    db = read_fasta.read_taxa_fasta(db_fasta)

    start = perf_counter()
    database = classifier.make_classifier(db)
    end = perf_counter()
    print(f"Classifer took {end-start:.2f} seconds to build database")

    start = perf_counter()
    classified = classifier.classify_sequences(test_seqs, database)
    end = perf_counter()
    print(f"Sequences classified in {end-start:.2f} seconds")

    print("Top 5 results:")
    classified = results.summarize_predictions(classified)
    print(classified.head())
