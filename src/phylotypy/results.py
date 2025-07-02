import pandas as pd


def summarize_predictions(classified: dict | pd.DataFrame, n_levels: int = 6):
    if isinstance(classified, dict):
        classified_df = pd.DataFrame(classified)
    else:
        classified_df = classified.copy()
    taxa_levels_full = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
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


def prevalence(data: pd.Series, threshold: float = 10):
    # Determine if the counts data for a specific sequence is above a threshold value
    filtered = data[data > threshold]
    prev = len(filtered) / len(data)
    return int(100 * prev)


if __name__ == "__main__":
    print(f"{__name__}")
