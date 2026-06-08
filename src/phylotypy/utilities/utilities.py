import io
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path
import pickle
import re
import subprocess
import requests
import xml.etree.ElementTree as ET
import pandas as pd


MAX_RETRIES = 5  # Limit the total number of attempts
INITIAL_WAIT_TIME = 2  # Start with a 2-second wait


def dataframe_to_fasta(df, fasta_file):
    with open(fasta_file, "w") as f:
        for index, row in df.iterrows():
            f.write(f">{row['id']}\n{row['sequence']}\n")


def taxa_to_dataframe(taxa_file):
    file_path = Path(taxa_file)
    if not file_path.is_file():
        print("Taxa file not found")
        return None
    taxa_table_df = pd.read_csv(file_path, sep="\t", names=["id", "taxonomy"])
    taxa_levels = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "_"]
    taxa_table_df[taxa_levels] = taxa_table_df['taxonomy'].str.split(';', expand=True)
    taxa_table_df['taxonomy'] = taxa_table_df['taxonomy'].str.rstrip(';')
    taxa_table_df.drop(["_"], axis="columns", inplace=True)
    return taxa_table_df


def fix_qiime_taxa(taxa_string):
    taxonomy = re.sub(r"\s*\w+__", "", taxa_string)
    return taxonomy


def convert_dict_keys_to_int(d):
    """
    Convert the keys of a dictionary from strings to integers.

    Parameters:
    d (dict): The dictionary with string keys to be converted.

    Returns:
    dict: A new dictionary with integer keys.
    """
    return {int(key): value for key, value in d.items()}


def pickle_and_compress(obj, output_file: str | Path):
    """
    Pickles a Python object and compresses it into a .pkl.gz file using gzip or pigz.

    Parameters:
        obj (object): The Python object to pickle and compress.
        output_file (str): The path to the output compressed .pkl.gz file.
    """
    # Start the subprocess to compress the output with pigz (parallel gzip)
    with subprocess.Popen(['pigz', '-c'], stdin=subprocess.PIPE, stdout=open(output_file, 'wb')) as proc:
        # Create an in-memory buffer to pickle the object
        with io.BytesIO() as buffer:
            pickle.dump(obj, buffer)  # Pickle the object into the buffer
            buffer.seek(0)  # Go to the start of the buffer before writing
            proc.stdin.write(buffer.read())  # Write the pickled object to the subprocess for compression


def unpickle_and_decompress(input_file: str | Path):
    """
    Unpickles and decompresses a .pkl.gz file to retrieve the original Python object.

    Parameters:
        input_file (str): The path to the compressed .pkl.gz file.

    Returns:
        object: The Python object that was pickled and compressed.
    """
    # Open the gzipped file and unpickle the object
    with gzip.open(input_file, 'rb') as f:
        obj = pickle.load(f)  # Unpickle and load the object from the file
    return obj


def summarize_taxa_ids(api_res):
    '''Formats the NCBI esummary "result" output'''
    recs = defaultdict(list)
    results = api_res.get("result", {})
    for kk, vv in results.items():
        if "uids" not in kk:
            for k, v in vv.items():
                recs[k].append(v)
    return recs


def get_eutils_results(url, payload):
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=payload)
            r.raise_for_status()
            
            fmt = payload.get("retmode", None)
            if fmt == "json":
                try:
                    return r.json()
                except json.JSONDecodeError:
                    print(f"Request failed to decode {url}")
                    return r.text
            return r.text
            
        except requests.exceptions.HTTPError as e:
            failed_response = e.response
            if failed_response.status_code == 429:
                wait_time = INITIAL_WAIT_TIME * (2 ** attempt)
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    wait_time = int(retry_after)
                if attempt +1 == MAX_RETRIES:
                    print(f"Max retries reached for {url}")
                    raise
                print(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                raise
                
        except requests.exceptions.Timeout:
            print("Request timed out")
            raise
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed {e}")
            raise


def parse_taxa_xml(res):
    root = ET.fromstring(res)
    top_level = root.findall("Taxon")
    
    def parse_lineage(lineage):
        saved_lineage = {}
        for rec in lineage:
            level = rec.findtext("Rank")
            taxon = rec.findtext("ScientificName")
            saved_lineage[level] = taxon
        return saved_lineage

    all_results = []
    for taxon in top_level:
        taxid = taxon.findtext("TaxId")
        name = taxon.findtext("ScientificName")
        rank = taxon.findtext("Rank")
        
        extended_lineage = parse_lineage(taxon.find("LineageEx"))
        extended_lineage["taxid"] = taxid
        extended_lineage[rank] = name
        all_results.append(extended_lineage)
    return all_results


def get_taxa_ids(taxa_names):
    if not isinstance(taxa_names, list):
        taxa_names = [taxa_names]
    terms = " OR ".join(taxa_names)
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    payload = dict(db="taxonomy", term=terms, retmode="json")
    response = get_eutils_results(url=search_url, payload=payload)

    exceed_limit = response.get("error", [])
    if exceed_limit:
        print(exceed_limit)
        return dict(error=exceed_limit)

    ids_list = response.get("esearchresult", {}).get("idlist", [])
    taxa_errors = response.get("esearchresult", {}).get("errorlist", [])
    if taxa_errors:
        print("Taxa not found: ", taxa_errors)

    taxa_id_payload = dict(db="taxonomy", id=",".join(ids_list), retmode="xml")
    found_ids = get_eutils_results(url=fetch_url, payload=taxa_id_payload)

    return found_ids


if __name__ == "__main__":
    print("Support tools for phylotypy package")
    taxa_list = [
        "Methanobrevibacter",
        "Halobacterium",
        "Nitrosopumilus",
        "Thermoproteus",
        "Sulfolobus",
        "Escherichia",
        "Staphylococcus",
        "Bacillus",
        "Test",
        "Pseudomonas",
        "Salmonella"
    ]
    res0 = get_taxa_ids(taxa_list)
    res0_dict = parse_taxa_xml(res0)
    # res0_dict = summarize_taxa_ids(res0)
    res0_df = pd.DataFrame(res0_dict)
    print(res0_df.head())
    # print(get_taxa_ids(["Methanobrevibacter"]))



