from pathlib import Path

from phylotypy import classifier, results
from phylotypy.utilities import read_fasta


if __name__ == "__main__":
    rdp_fasta = Path("data/rdp_16S_v19.dada2.fasta")

    moving_pics = read_fasta.read_taxa_fasta("data/dna_moving_pictures.fasta")

    if rdp_fasta.parent / "database.pkl":
        database = classifier.load_classifier(rdp_fasta.parent / "database.pkl")
    else:
        database = classifier.make_classifier(rdp_fasta, rdp_fasta.parent)

    classified = classifier.classify_sequences(moving_pics, database)

    classified = results.summarize_predictions(classified)

    print(classified.head())
