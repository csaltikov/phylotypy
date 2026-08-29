import pandas as pd
import numpy as np


def summarize_predictions(classified: dict | pd.DataFrame, n_levels: int | None = None):
    """Expand phylotypy classifications into per-rank columns and lineage strings.

    Takes the output of a phylotypy classification, where the ``classification``
    column holds semicolon-delimited taxa annotated with bootstrap confidence
    values in parentheses, e.g. 'Bacteria(100);Pseudomonadota(100);Gammaproteobacteria(100);Chromatiales(100)....'
    and splits it into one column per taxonomic rank.
    Two convenience columns are also built: ``observed`` (the cleaned taxa
    rejoined with ``;``) and ``lineage`` (the same taxa prefixed with
    greengenes-style rank codes such as ``k__``, ``p__``, ``c__``, ...).

    The number of ranks is detected from the data by default, using the most
    common number of semicolon separators across all rows. If ``n_levels`` is
    given and disagrees with the detected value, a message is printed and the
    detected value is used. Depth is assumed to be consistent, which holds when
    the input comes from ``classify_sequences``: ``print_taxonomy`` pads every
    consensus out to a fixed ``n_levels``, backfilling ranks dropped by
    ``filter_taxonomy`` as ``<last_taxon>_unclassified``.

    Confidence filtering has already been applied upstream by
    ``filter_taxonomy`` (via ``min_confidence``), so this function does no
    confidence-based rewriting. Any ``_sp`` or ``_unclassified`` suffix present
    in the classification is preserved as-is.

    Parameters
    ----------
    classified : dict or pandas.DataFrame
        Classification results containing a ``classification`` column. A dict
        is converted via ``pandas.DataFrame``.
    n_levels : int, optional
        Number of taxonomic ranks to expand, from Kingdom down to Species
        (max 7). If ``None`` (default), the value is auto-detected from the
        data.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with one column per taxonomic rank plus ``observed``
        and ``lineage`` columns. Confidence annotations like ``(100)`` are
        stripped from the rank columns.
    """
    if isinstance(classified, dict):
        classified_df = pd.DataFrame(classified)
    else:
        classified_df = classified.copy()
    
    taxa_levels_full = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    
    counts = classified["classification"].str.count(";")
    n_levels_majority = int(np.bincount(counts).argmax()) + 1

    if n_levels is None:
        n_levels = n_levels_majority
    elif n_levels != n_levels_majority:
        print(f"Levels detected {n_levels_majority} do not match specified levels {n_levels}")
        n_levels = n_levels_majority

    taxa_levels = taxa_levels_full[:n_levels]
    tax_level_codes = [f"{t[0].lower()}__" for t in taxa_levels]
    
    classified_df[taxa_levels] = classified_df["classification"].str.split(";", expand=True)

    def join_taxa(taxa_split):
        """Prefix each taxon with its greengenes-style rank code and join with ';'."""
        return ";".join([f'{tax_level_codes[i]}{tax}' for i, tax in enumerate(taxa_split)])

    def remove_confidence(col):
        """Strip trailing bootstrap confidence annotations like '(100)' from a column."""
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
