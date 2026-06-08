import pandas as pd
import numpy as np


def summarize_predictions(classified: dict | pd.DataFrame, n_levels: int = 6):
    if isinstance(classified, dict):
        classified_df = pd.DataFrame(classified)
    else:
        classified_df = classified.copy()
    
    taxa_levels_full = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    
    counts = classified["classification"].str.count(";")
    majority = np.bincount(counts).argmax()
    
    n_levels = int(majority) + 1
    taxa_levels = taxa_levels_full[:n_levels]
    tax_level_codes = [f"{t[0].lower()}__" for t in taxa_levels]
    
    classified_df[taxa_levels] = classified_df["classification"].str.split(";", expand=True)

    def join_taxa(taxa_split):
        return ";".join([f'{tax_level_codes[i]}{tax}' for i, tax in enumerate(taxa_split)])

    def remove_confidence(col):
        return col.str.replace(r"\(\d+\)", "", regex=True)

    classified_df[taxa_levels] = classified_df[taxa_levels].apply(remove_confidence)
    classified_df["observed"] = classified_df[taxa_levels].apply(lambda row: ';'.join(row.values), axis=1)
    classified_df["lineage"] = classified_df[taxa_levels].apply(lambda row: join_taxa(row.values), axis=1)
    return classified_df


def prevalence(data: pd.Series, min_count: int = 1) -> int:
    '''Percent of samples in which a feature meets the minimum count threshold.
    
    Args:
        data: pd.Series of the features and their counts across all samples
        min_count: the minimum counts required to establish a sample positive for a feature
    
    Returns:
        int: prevalence as a whole-number percentage (0–100)
    '''
    return int(100 * (data >= min_count).sum() / len(data))


if __name__ == "__main__":
    print(f"{__name__}")
