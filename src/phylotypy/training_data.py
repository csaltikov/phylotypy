from pathlib import Path
import pandas as pd
import requests
import tarfile
import numpy as np

from phylotypy.utilities import read_fasta


def download_and_extract(url, output_dir: str | Path):
    """
    Downloads a tar.gz file from the given URL and extracts it to the specified output directory.

    Parameters:
    url (str): The URL of the tar.gz file to download.
    output_dir (str or Path): The directory where the contents should be extracted.
    """
    # Ensure output_dir is a Path object
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Path to the downloaded file
    download_path = output_dir / Path(url).name.rstrip("?download")

    print(f"The file was downloaded: {download_path.exists()}")

    if not download_path.exists():
        print("Downloading the file...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(download_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"Downloaded: {download_path}")
        else:
            print("Failed to download the file")
            response.raise_for_status()

    print(f"File downloaded to {output_dir}")


def rdp_train_set_19(out_dir: str | Path):
    """
    Download the RDP's latest 16S rRNA full length gene training data set.
    The trainset is trainset19_072023.  The function converts the files into
    a Pandas Dataframe for use in the Phylotypy.train() function
    """
    print("Starting...")
    link_address = "https://mothur.s3.us-east-2.amazonaws.com/wiki/trainset19_072023.rdp.tgz"
    download_and_extract(link_address, out_dir)
    fasta_file = Path(out_dir) / "trainset19_072023.rdp.tgz"
    with tarfile.open(fasta_file, 'r:gz') as tar:
        tar.extractall(filter="data")
        for item in tar.getmembers():
            print(f"Extracted: {item.name}")


def open_training_set(out_dir: str | Path, fasta_file: str | Path, db_name: str):
    if "rdp" in fasta_file:
        db_name = "trainset19_072023_db.csv"
    refdb = read_fasta.read_taxa_fasta(fasta_file)
    db_file_path = Path(out_dir) / db_name  # "trainset19_072023_db.csv"
    refdb.to_csv(db_file_path, index=False)
    print("Done processing fasta file")
    print(f"trainset file is located at {db_file_path}")
    return refdb


def silva_train_set(out_dir):
    out_path = Path(out_dir)
    print("Starting...file is big!")
    link_address = "https://zenodo.org/records/3986799/files/silva_nr99_v138_train_set.fa.gz?download"
    download_and_extract(link_address, out_path)
    fasta_file = out_path.joinpath("silva_nr99_v138_train_set.fa.gz")
    ref_db: pd.DataFrame = read_fasta.read_taxa_fasta(fasta_file)
    print("Done processing fasta file")
    silva_out = out_path.joinpath("silva_nr99_v138_train_set.parquet")

    chunk_size = 10000

    for i, chunk in enumerate(np.array_split(ref_db, ref_db.shape[0] // chunk_size)):
        chunk.to_parquet(silva_out, compression='snappy', engine='pyarrow', index=False)


def filter_train_set(df: pd.DataFrame, n_levels: int = 6, **kwargs) -> pd.DataFrame:
    """
    Filter a reference database by taxonomy level and remove noisy sequences.

    Removes sequences containing unwanted taxonomic terms (e.g. metagenomes,
    Eukaryota, Candidatus) and retains only sequences with a specific number
    of taxonomic levels. Useful for standardizing training data before
    building a classifier.

    Args:
        df: DataFrame with 'id' and 'sequence' columns where 'id' contains
            semicolon-delimited taxonomy strings e.g.
            'Bacteria;Firmicutes;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus'
        **kwargs:
            terms (str): Pipe-delimited regex pattern of terms to exclude.
                Default: 'Incertae|Sedis|metagenome|Eukaryota|Metagenome|Candidatus|culture'
            n_levels (int): Number of taxonomic levels to retain. Default: 6

    Returns:
        pd.DataFrame: Filtered DataFrame with only sequences matching the
            specified taxonomy depth and free of noisy terms.

    Examples:
        >>> from phylotypy import read_fasta, training_data
        >>> silva = read_fasta.read_taxa_fasta("silva_138.fasta.gz")
        >>> silva.shape
        (231218, 2)

        >>> # Default filtering — 6 levels, standard noise terms
        >>> filtered = training_data.filter_train_set(silva)
        >>> filtered.shape
        (158970, 3)

        >>> # Custom taxonomy depth
        >>> filtered = training_data.filter_train_set(silva, n_levels=7)

        >>> # Add custom noise terms
        >>> filtered = training_data.filter_train_set(silva,
        ...                                           terms="Incertae|metagenome|Eukaryota|uncultured")
    """
    noise = "Incertae|Sedis|metagenome|Eukaryota|Metagenome|Candidatus|culture|endosymbiont"
    
    taxa_levels_full = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    terms = kwargs.get("terms", noise)
    threshold = kwargs.get("threshold", 5)
    
    df_ = (df[~df["id"].str.contains(terms, na=False)]
           .assign(levels=lambda x: x['id'].str.count(";") + 1)
           )
    df_["id"] = df_["id"].str.replace(r"[\[\]]", "", regex=True)
    
    if n_levels == 7:
        df_ = df_[df_["levels"] >= 6].copy()
        df_[taxa_levels_full] = df_['id'].str.split(";", expand=True)
        df_["Species"] = df_["Species"].fillna("")
        
        # fill_species
        df_["Species"] = np.where(
            df_["Species"].str.strip() == "",
            df_["Genus"] + "_sp",
            df_["Genus"] + "_" + df_["Species"]
        )
        
        # collapse low-rep taxa
        species_counts = df_["Species"].value_counts()
        low_rep_taxa = set(species_counts[species_counts < threshold].index)  # set for O(1) lookup
        
        df_["Species"] = np.where(
            df_["Species"].isin(low_rep_taxa),
            df_["Genus"] + "_sp",
            df_["Species"]
        )
        
        df_["id"] = df_[taxa_levels_full].fillna("").apply(lambda x: ";".join(x), axis=1)
        
        post_collapse_counts = df_["id"].value_counts()
        still_singleton = set(post_collapse_counts[post_collapse_counts < 2].index)
        df_ = df_[~df_["id"].isin(still_singleton)]
        
        return df_[["id", "sequence"]]
    else:
        return df_[df_["levels"] == n_levels]


def down_sample(df, col="id", n=200, random_state=None) -> pd.DataFrame:
    data = []
    for _, group in df.groupby(col):
        if len(group) > n:
            data.append(group.sample(n, random_state=random_state))
        else:
            data.append(group)
    return pd.concat(data, ignore_index=True)


if __name__ == "__main__":
    print("Tools for processing phylotypy training data sets")
