#!/usr/bin/env python3
"""Command-line interface for phylotypy.

Two subcommands:
  phylotypy build     -- build a classifier database from a reference fasta and save it (pickle)
  phylotypy classify  -- classify a fasta of sequences against a database

`classify --db` accepts either a prebuilt classifier (.pkl/.pickle, from `build`)
or a raw reference fasta -- in the latter case the classifier is built on the fly
before classifying (slower; use --save-db to cache it for next time).
"""
import argparse
import sys
import time
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

try:
    __version__ = version("phylotypy")
except PackageNotFoundError:
    __version__ = "unknown"


def _existing_file(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"file not found: {value}")
    return path


def _add_common_classify_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kmer-size", type=int, default=8,
                        help="k-mer size; must match the database's k-mer size (default: 8)")
    parser.add_argument("--num-bootstrap", type=int, default=100,
                        help="number of bootstrap replicates for the confidence estimate (default: 100)")
    parser.add_argument("--min-consensus", type=float, default=80,
                        help="bootstrap confidence threshold 0-100; ranks below this are "
                             "reported as '_unclassified' (default: 80)")
    parser.add_argument("--n-levels", type=int, default=None,
                        help="number of taxonomic levels in the output (default: auto-detected "
                             "from the reference database)")
    parser.add_argument("--threads", type=int, default=None,
                        help="number of CPU threads to use for classification (default: all cores)")
    parser.add_argument("--force", action="store_true",
                        help="skip the memory-safety check before bootstrap sampling")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print progress messages")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phylotypy",
        description="Naive Bayes classifier for 16S rRNA sequence data.",
    )
    parser.add_argument("--version", action="version", version=f"phylotypy {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- build ---
    build_cmd = subparsers.add_parser(
        "build",
        help="build a classifier database from a reference fasta and save it to disk",
    )
    build_cmd.add_argument("-i", "--input", required=True, type=_existing_file,
                           help="reference fasta with taxonomy strings in the sequence headers")
    build_cmd.add_argument("-o", "--out", required=True, type=Path,
                           help="output path for the pickled classifier database")
    build_cmd.add_argument("--kmer-size", type=int, default=8,
                           help="k-mer size used to build the database (default: 8)")
    build_cmd.add_argument("--threads", type=int, default=4,
                           help="number of CPU processes to use while building (default: 4)")
    build_cmd.add_argument("--filter-db", action="store_true",
                           help="filter the reference database to a single consistent "
                                "taxonomic depth and drop noisy entries before building")
    build_cmd.add_argument("--max-per-genus", type=int, default=None,
                           help="down-sample to at most this many sequences per genus "
                                "(requires --filter-db)")
    build_cmd.add_argument("--n-levels", type=int, default=None,
                           help="taxonomic depth to keep when --filter-db is set "
                                "(default: auto-detected)")
    build_cmd.add_argument("-v", "--verbose", action="store_true",
                           help="print progress messages")
    build_cmd.set_defaults(func=run_build)

    # --- classify ---
    classify_cmd = subparsers.add_parser(
        "classify",
        help="classify sequences against a database",
    )
    classify_cmd.add_argument("-i", "--input", required=True, type=_existing_file,
                              help="fasta of representative/dereplicated sequences to classify "
                                   "(e.g. ASVs or OTU centroids -- not raw reads)")
    classify_cmd.add_argument("-d", "--db", required=True, type=_existing_file,
                              help="reference database: a prebuilt classifier (.pkl/.pickle, "
                                   "from `phylotypy build`) or a raw reference fasta, which "
                                   "will be built on the fly")
    classify_cmd.add_argument("-o", "--out", required=True, type=Path,
                              help="output path for classification results (.tsv or .csv; "
                                   "default tab-separated)")
    classify_cmd.add_argument("--save-db", type=Path, default=None,
                              help="if --db is a raw fasta, add a file path to save the built "
                                   " classifier here for reuse (avoids rebuilding it on the next run)")
    classify_cmd.add_argument("--res-extended", help="Created an extended results report "
                              "including qiime formatted lineage, lineages split into taxonomic levels",
                              action="store_true")
    _add_common_classify_args(classify_cmd)
    classify_cmd.set_defaults(func=run_classify)

    return parser


def _is_pickled_db(path: Path) -> bool:
    return path.suffix.lower() in {".pkl", ".pickle"}


def run_build(args: argparse.Namespace) -> None:
    from phylotypy import classifier

    t0 = time.time()
    database = classifier.make_classifier(
        args.input,
        kmer_size=args.kmer_size,
        multiprocess=args.threads > 1,
        n_cpu=args.threads,
        verbose=args.verbose,
        filter_db=args.filter_db,
        max_per_genus=args.max_per_genus,
        n_levels=args.n_levels,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(args.out, "wb") as f:
        pickle.dump(database, f)

    print(f"Saved classifier database to {args.out} ({time.time() - t0:.1f}s)")


def run_classify(args: argparse.Namespace) -> None:
    from phylotypy import classifier, results

    if args.threads:
        import numba
        numba.set_num_threads(args.threads)

    if _is_pickled_db(args.db):
        database = classifier.load_classifier(args.db)
    else:
        if args.verbose:
            print(f"{args.db} doesn't look like a prebuilt database (.pkl/.pickle); "
                  "building a classifier from it first...")
        database = classifier.make_classifier(
            args.db,
            kmer_size=args.kmer_size,
            multiprocess=(args.threads or 1) > 1,
            n_cpu=args.threads or 4,
            verbose=args.verbose,
        )
        if args.save_db:
            args.save_db.parent.mkdir(parents=True, exist_ok=True)
            import pickle
            with open(args.save_db, "wb") as f:
                pickle.dump(database, f)
            print(f"Saved built classifier database to {args.save_db}")

    t0 = time.time()
    classified_results = classifier.classify_sequences(
        args.input,
        database,
        verbose=args.verbose,
        min_confidence=args.min_consensus,
        n_levels=args.n_levels,
        kmer_size=args.kmer_size,
        num_bs=args.num_bootstrap,
        force=args.force,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sep = "," if args.out.suffix.lower() == ".csv" else "\t"
    if args.res_extended:
        classified_results = results.summarize_predictions(classified_results, n_levels=args.n_levels)
    classified_results.to_csv(args.out, sep=sep, index=False)

    print(f"Classified {len(classified_results)} sequences in {time.time() - t0:.1f}s -> {args.out}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, MemoryError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
