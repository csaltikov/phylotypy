from pathlib import Path
import requests
import tarfile
import gzip

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


def rdp_train_set_19():
    """
    Download the RDP's latest 16S rRNA full length gene training data set.
    The trainset is trainset19_072023.  The function converts the files into
    a Pandas Dataframe for use in the Phylotypy.train() function
    """
    print("Starting...")
    link_address = "https://mothur.s3.us-east-2.amazonaws.com/wiki/trainset19_072023.rdp.tgz"
    download_and_extract(link_address, "training_data")
    fasta_file = "training_data/rdp_16S_v19.dada2.fasta"
    refdb = read_fasta.read_taxa_fasta(fasta_file)
    db_file_path = Path("data/trainset19_072023_db.csv")
    refdb.to_csv(db_file_path, index=False)
    print("Done processing fasta file")
    print(f"trainset file is located at {db_file_path}")


def silva_train_set(out_dir):
    out_path = Path(out_dir)
    print("Starting...file is big!")
    link_address = "https://zenodo.org/records/3986799/files/silva_nr99_v138_train_set.fa.gz?download"
    download_and_extract(link_address, out_path)
    fasta_file = out_path.joinpath("silva_nr99_v138_train_set.fa.gz")
    ref_db = read_fasta.read_taxa_fasta(fasta_file)
    print("Done processing fasta file")
    silva_out = out_path.joinpath("silva_nr99_v138_train_set.parquet")

    chunk_size = 10000

    for i, chunk in enumerate(np.array_split(ref_db, ref_db.shape[0] // chunk_size)):
        mode = "w" if i == 0 else "a"
        chunk.to_parquet(silva_out, compression='snappy', engine='pyarrow', index=False)


if __name__ == "__main__":
    rdp_train_set_19()
