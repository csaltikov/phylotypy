from pathlib import Path
from phylotypy import classifier, results, kmers
from phylotypy import read_fasta
from time import perf_counter


if __name__ == "__main__":
    from time import perf_counter

    db_fasta = Path("/Users/caltikov/PycharmProjects/phylotypy_data/local_data/silva/silva_138.2_16Sprok_dada2.fasta.gz")
    print("Loading test sequences...")
    test_seqs = read_fasta.read_taxa_fasta("data/dna_moving_pictures.fasta")

    print("Loading database sequences...")
    db = read_fasta.read_taxa_fasta(db_fasta)

    start = perf_counter()
    database = classifier.make_classifier(db)
    end = perf_counter()
    print(f"Classifer make database took {end-start:.2f} seconds to build")

    start = perf_counter()
    classified = classifier.classify_sequences(test_seqs, database)
    end = perf_counter()
    print(f"Classifying sequences took {end-start:.2f} seconds")

    print("Top 5 results:")
    classified = results.summarize_predictions(classified)
    print(classified.head())
