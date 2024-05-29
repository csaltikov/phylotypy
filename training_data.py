from pathlib import Path
import requests
import tarfile

import utilities


def download_and_extract(url, output_dir):
    """
    Downloads a tar.gz file from the given URL and extracts it to the specified output directory.

    Parameters:
    url (str): The URL of the tar.gz file to download.
    output_dir (str or Path): The directory where the contents should be extracted.
    """
    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Path to the downloaded file
    download_path = output_dir / Path(url).name

    # Download the file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(download_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
    else:
        print("Failed to download the file")
        response.raise_for_status()

    # Extract the tar.gz file
    with tarfile.open(download_path, 'r:gz') as tar:
        tar.extractall(path=output_dir)

    # Optionally, delete the downloaded tar.gz file to save space
    download_path.unlink()

    print(f"File downloaded and extracted to {output_dir}")


def rdp_train_set_19():
    """
    Download the RDP's latest 16S rRNA full length gene training data set.
    The trainset is trainset19_072023.  The function converts the files into
    a Pandas Dataframe for use in the Phylotypy.train() function
    """
    print("Starting...")
    link_address = "https://mothur.s3.us-east-2.amazonaws.com/wiki/trainset19_072023.rdp.tgz"
    download_and_extract(link_address, "training_data")
    fasta_file = "training_data/trainset19_072023.rdp.fasta"
    taxa_file = "training_data/trainset19_072023.rdp.tax"
    refdb = utilities.fasta_to_dataframe_taxa(fasta_file, taxa_file)
    refdb.to_csv(Path("data/trainset19_072023_db.csv"), index=False)
    print("Done processing fasta file")


def silva_trainset():
    print("Starting...file is big!")
    link_address = "https://zenodo.org/records/3986799/files/silva_nr99_v138_train_set.fa.gz?download"
    download_and_extract(link_address, "training_data")
    fasta_file = "training_data/silva_nr99_v138_train_set.fa"
    ref_db = utilities.fasta_to_dataframe(fasta_file)
    print("Done processing fasta file")
    ref_db.to_csv("data/silva_nr99_v138_train_set.csv")



if __name__ == "__main__":
    rdp_train_set_19()
