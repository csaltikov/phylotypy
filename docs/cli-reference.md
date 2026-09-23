# Command Line Interface (CLI) Reference

## Building and reusing a classifier database

You can save the classifier (a pickle file) and reuse it later by specifying `--save-db`:

```shell
phylotypy classify --input dna_moving_pictures.fasta \
                   --db rdp_16S_v19.dada2.fasta \
                   --save-db rdp_classifer.pickle \ # set the path
                   --out classied_seqs.tsv \
                   --verbose

# resuse on another run saves time since the db is already built
phylotypy classify --input my_sequences.fasta \
                   --db rdp_classifer.pickle \
                   --out classied_my_seqs.tsv \
                   --verbose
```

Or build the database up front with `phylotypy build` and pass it to `classify` later — see the `phylotypy build --help` output below.

## Terminal report

Add `--terminal-report` (or the shorter `--t-report`) to print a bar chart of the
classifications right after the results file is written. It summarizes at the
phylum level by default:

```shell
phylotypy classify --input dna_moving_pictures.fasta \
                   --db rdp_classifer.pickle \
                   --out classified_seqs.tsv \
                   --t-report
```

```text
Phylum-level summary  770 sequences · 20 taxa (19 named)
────────────────────────────────────────────────────────────────────────────────
Bacillota              ████████████████████████████████████████████  277  36.0%
Pseudomonadota         ███████████████████████▉                      150  19.5%
Bacteroidota           █████████████████████▊                        137  17.8%
Bacteria_unclassified  ██████████▋                                    67   8.7%
Actinomycetota         ██████████                                     63   8.2%
Fusobacteriota         ███▋                                           23   3.0%
Cyanobacteriota        █▍                                              9   1.2%
Campylobacterota       █▏                                              7   0.9%
Verrucomicrobiota      █                                               6   0.8%
Plantae                █                                               6   0.8%
Spirochaetota          █                                               6   0.8%
Mycoplasmatota         ▌                                               3   0.4%
Planctomycetota        ▌                                               3   0.4%
Acidobacteriota        ▌                                               3   0.4%
Synergistota           ▍                                               2   0.3%
Other (5 taxa)         █▎                                              8   1.0%
────────────────────────────────────────────────────────────────────────────────
resolved at phylum: 703/770 (91.3%) · mean confidence 98.4
```

Bars are scaled to the longest row drawn, the `Other` row included, so a folded
tail that outweighs every individual taxon is shown as the largest bar rather
than being clamped to the top row's length — this matters at genus level and
below, where the tail is usually dominant.

The counts are numbers of sequences (ASVs/OTUs), **not read abundances**: a
phylum with 277 ASVs is 36% of your distinct sequences, not necessarily 36% of
your community. `*_unclassified` rows are drawn dim, and are excluded from the
"named" count in the header and from "resolved at <rank>" in the footer.

Use `--report-rank` for a different rank and `--report-top` to change how many
taxa are charted before the tail is folded into an "Other" row
(`--report-top 0` charts everything):

```shell
phylotypy classify -i asvs.fasta -d rdp_classifer.pickle -o classified.tsv \
                   --t-report --report-rank genus --report-top 25
```

The same report is available from the API, for example when working in a REPL:

```python
from phylotypy import classifier, terminal_report

classified = classifier.classify_sequences(seqs, database)
terminal_report.print_report(classified, rank="phylum", top=20)

# or capture it as a string (e.g. to write into a log)
text = terminal_report.render_report(classified, rank="genus", color=False, width=100)

# just the numbers, no chart
summary = terminal_report.summarize_rank(classified, "phylum")
summary.counts.head()
```

Color is used only when stdout is a terminal, and is disabled when `NO_COLOR` is
set, so piping or redirecting the output stays clean.

## `phylotypy report`

If you already have a classified results file (from `phylotypy classify`, with or
without `--t-report`), `phylotypy report` prints the same bar-chart summary from
that file directly, without re-classifying:

```shell
phylotypy report --input classified_seqs.tsv
```

```text
Phylum-level summary  770 sequences · 20 taxa (19 named)
────────────────────────────────────────────────────────────────────────────────
Bacillota              ████████████████████████████████████████████  277  36.0%
Pseudomonadota         ███████████████████████▉                      150  19.5%
Bacteroidota           █████████████████████▊                        137  17.8%
Bacteria_unclassified  ██████████▋                                    67   8.7%
Actinomycetota         ██████████                                     63   8.2%
Fusobacteriota         ███▋                                           23   3.0%
Cyanobacteriota        █▍                                              9   1.2%
Campylobacterota       █▏                                              7   0.9%
Verrucomicrobiota      █                                               6   0.8%
Plantae                █                                               6   0.8%
Spirochaetota          █                                               6   0.8%
Mycoplasmatota         ▌                                               3   0.4%
Planctomycetota        ▌                                               3   0.4%
Acidobacteriota        ▌                                               3   0.4%
Synergistota           ▍                                               2   0.3%
Other (5 taxa)         █▎                                              8   1.0%
────────────────────────────────────────────────────────────────────────────────
resolved at phylum: 703/770 (91.3%) · mean confidence 98.4
```

It accepts the same `--report-rank` and `--report-top` options as `--t-report`:

```shell
phylotypy report -i classified_seqs.tsv --report-rank genus --report-top 25
```

```shell
phylotypy report --help

usage: phylotypy report [-h] -i INPUT [--report-rank REPORT_RANK]
                        [--report-top REPORT_TOP]

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     csv/tsv of the classified results
  --report-rank REPORT_RANK
                        taxonomic rank to summarize: kingdom, phylum, class,
                        order, family, genus or species (default: phylum)
  --report-top REPORT_TOP
                        maximum number of taxa to chart; the rest are folded
                        into an 'Other' row, 0 for no limit (default: 15)
```

## Help menu

```shell
phylotypy --help

usage: phylotypy [-h] [--version] {build,classify,report} ...

Naive Bayes classifier for 16S rRNA sequences: classifies ASVs/OTUs from
DADA2, QIIME2, or raw FASTA files against a reference database (e.g. RDP,
Silva).

positional arguments:
  {build,classify,report}
    build               build a classifier database from a reference fasta and save it to disk
    classify            classify sequences against a database
    report              create a taxon summary chart on the command line

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit

Run 'phylotypy <command> --help' for options specific to that command (e.g.
'phylotypy classify --help').
```
--
```shell
phylotypy classify --help
usage: phylotypy classify [-h] -i INPUT -d DB -o OUT [--save-db SAVE_DB] [--res-extended] [--terminal-report] [--report-rank REPORT_RANK]
                          [--report-top REPORT_TOP] [--kmer-size KMER_SIZE] [--num-bootstrap NUM_BOOTSTRAP] [--min-consensus MIN_CONSENSUS]
                          [--n-levels N_LEVELS] [--threads THREADS] [--force] [-v]

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     fasta of representative/dereplicated sequences to classify (e.g. ASVs or OTU centroids -- not raw reads)
  -d, --db DB           reference database: a prebuilt classifier (.pkl/.pickle, from `phylotypy build`) or a raw reference fasta, which will be built on the fly
  -o, --out OUT         output path for classification results (.tsv or .csv; default tab-separated)
  --save-db SAVE_DB     if --db is a raw fasta, add a file path to save the built classifier here for reuse (avoids rebuilding it on the next run)
  --res-extended        Created an extended results report including qiime formatted lineage, lineages split into taxonomic levels
  --terminal-report, --t-report
                        print a bar-chart summary of the classifications to the terminal after writing the results
  --report-rank REPORT_RANK
                        taxonomic rank to summarize with --terminal-report: kingdom, phylum, class, order, family, genus or species (default: phylum)
  --report-top REPORT_TOP
                        maximum number of taxa to chart with --terminal-report; the rest are folded into an 'Other' row, 0 for no limit (default: 15)
  --kmer-size KMER_SIZE
                        k-mer size; must match the database's k-mer size (default: 8)
  --num-bootstrap NUM_BOOTSTRAP
                        number of bootstrap replicates for the confidence estimate (default: 100)
  --min-consensus MIN_CONSENSUS
                        bootstrap confidence threshold 0-100; ranks below this are reported as '_unclassified' (default: 80)
  --n-levels N_LEVELS   number of taxonomic levels in the output (default: auto-detected from the reference database)
  --threads THREADS     number of CPU threads to use for classification (default: all cores)
  --force               skip the memory-safety check before bootstrap sampling
  -v, --verbose         print progress messages
```
--
```shell
phylotypy build --help
usage: phylotypy build [-h] -i INPUT -o OUT [--kmer-size KMER_SIZE] [--threads THREADS] [--filter-db] [--max-per-genus MAX_PER_GENUS] [--n-levels N_LEVELS] [-v]

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     reference fasta with taxonomy strings in the sequence headers
  -o, --out OUT         output path for the pickled classifier database
  --kmer-size KMER_SIZE
                        k-mer size used to build the database (default: 8)
  --threads THREADS     number of CPU processes to use while building (default: 4)
  --filter-db           filter the reference database to a single consistent taxonomic depth and drop noisy entries before building
  --max-per-genus MAX_PER_GENUS
                        down-sample to at most this many sequences per genus (requires --filter-db)
  --n-levels N_LEVELS   taxonomic depth to keep when --filter-db is set (default: auto-detected)
  -v, --verbose         print progress messages
```
