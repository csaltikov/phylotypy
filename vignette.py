from pathlib import Path
from phylotypy import classifier, results
from phylotypy import read_fasta


if __name__ == "__main__":
    from time import perf_counter

    db_fasta = Path("data/rdp_16S_v19.dada2.fasta")
    test_seqs = read_fasta.read_taxa_fasta("data/dna_moving_pictures.fasta")

    db = read_fasta.read_taxa_fasta(db_fasta)

    database = classifier.make_classifier(db)

    start = perf_counter()
    classified = classifier.classify_sequences(test_seqs, database)
    end = perf_counter()
    print(f"Finished in {end-start:.2f} seconds")

    classified = results.summarize_predictions(classified)
    print(classified.head())
