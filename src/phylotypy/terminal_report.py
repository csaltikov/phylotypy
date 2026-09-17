"""Terminal summary of a classification run.

Renders a horizontal bar chart of how many sequences were assigned to each
taxon at a given rank (phylum by default), plus a short header/footer with
counts, the fraction resolved at that rank, and mean bootstrap confidence.

Everything is drawn with Unicode block characters and (optionally) ANSI color,
so there is no extra dependency beyond what phylotypy already requires.

Examples:
    >>> from phylotypy import classifier, terminal_report
    >>> classified = classifier.classify_sequences(seqs, database)
    >>> terminal_report.print_report(classified, rank="phylum")
"""
from __future__ import annotations

import dataclasses
import os
import re
import shutil
import sys

import pandas as pd

TAXA_LEVELS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

RANK_ALIASES = {
    "domain": "Kingdom",
    "kingdom": "Kingdom",
    "phylum": "Phylum",
    "class": "Class",
    "order": "Order",
    "family": "Family",
    "genus": "Genus",
    "species": "Species",
}

#: Trailing bootstrap confidence annotation, e.g. "Pseudomonadota(97)".
_CONFIDENCE_RE = re.compile(r"\((\d+(?:\.\d+)?)\)\s*$")

_BLOCKS = " ▏▎▍▌▋▊▉█"  # 0/8 .. 8/8 of a cell

_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
}
# Bars cycle through this palette; unclassified rows are drawn dim instead.
_BAR_COLORS = ["\033[38;5;37m", "\033[38;5;68m", "\033[38;5;108m",
               "\033[38;5;173m", "\033[38;5;139m", "\033[38;5;179m"]


class ReportError(ValueError):
    """Raised when a report cannot be built from the given results."""


@dataclasses.dataclass
class RankSummary:
    """Per-taxon tallies for one taxonomic rank.

    Attributes:
        rank: Rank actually summarized (may be shallower than requested if the
            classifications do not reach that deep).
        requested_rank: Rank the caller asked for.
        counts: Sequence count per taxon, descending, unclassified taxa included.
        confidence: Mean bootstrap confidence per taxon at this rank, or None
            when the classification strings carry no confidence annotations.
        n_sequences: Total sequences summarized.
        n_unclassified: Sequences whose taxon at this rank is an
            ``*_unclassified`` placeholder.
    """
    rank: str
    requested_rank: str
    counts: pd.Series
    confidence: pd.Series | None
    n_sequences: int
    n_unclassified: int

    @property
    def n_taxa(self) -> int:
        """Number of distinct taxa at this rank, ``*_unclassified`` included."""
        return int(len(self.counts))

    @property
    def n_named_taxa(self) -> int:
        """Distinct taxa at this rank, excluding ``*_unclassified`` placeholders."""
        return int(sum(not is_unclassified(taxon) for taxon in self.counts.index))

    @property
    def n_classified(self) -> int:
        return self.n_sequences - self.n_unclassified

    @property
    def truncated(self) -> bool:
        """True when the requested rank was deeper than the data goes."""
        return self.rank != self.requested_rank


def is_unclassified(taxon: str) -> bool:
    return str(taxon).endswith("_unclassified")


def resolve_rank(rank: str | int) -> tuple[int, str]:
    """Map a rank name (or 0-based index) onto ``(index, canonical name)``."""
    if isinstance(rank, int) or (isinstance(rank, str) and rank.isdigit()):
        idx = int(rank)
        if not 0 <= idx < len(TAXA_LEVELS):
            raise ReportError(f"rank index out of range: {idx} (expected 0-{len(TAXA_LEVELS) - 1})")
        return idx, TAXA_LEVELS[idx]

    name = RANK_ALIASES.get(str(rank).strip().lower())
    if name is None:
        raise ReportError(
            f"unknown rank: {rank!r} (expected one of {', '.join(sorted(set(RANK_ALIASES)))})"
        )
    return TAXA_LEVELS.index(name), name


def split_rank(classification: pd.Series, index: int) -> tuple[pd.Series, pd.Series]:
    """Pull the taxon and its confidence at ``index`` out of lineage strings.

    Rows shallower than ``index`` yield NaN in both returned series.
    """
    fields = classification.astype(str).str.split(";")
    at_rank = fields.map(lambda parts: parts[index].strip() if len(parts) > index else None)

    confidence = at_rank.map(
        lambda taxon: float(m.group(1)) if taxon and (m := _CONFIDENCE_RE.search(taxon)) else None
    )
    taxa = at_rank.map(lambda taxon: _CONFIDENCE_RE.sub("", taxon).strip() if taxon else None)
    return taxa, confidence


def summarize_rank(classified: pd.DataFrame | dict,
                   rank: str | int = "Phylum",
                   *,
                   column: str = "classification") -> RankSummary:
    """Tally sequences per taxon at ``rank``.

    If the classifications are shallower than the requested rank, the deepest
    available rank is summarized instead and flagged on the returned summary.

    Args:
        classified: Output of ``classifier.classify_sequences`` (or a dict of
            the same shape) with a lineage column.
        rank: Rank name (e.g. "phylum", "genus") or 0-based index.
        column: Name of the lineage column. Defaults to "classification".

    Returns:
        RankSummary: counts, confidences and totals for that rank.
    """
    df = pd.DataFrame(classified) if isinstance(classified, dict) else classified
    if column not in df.columns:
        raise ReportError(f"no {column!r} column in the classification results")
    if df.empty:
        raise ReportError("no classification results to summarize")

    index, requested_name = resolve_rank(rank)
    depth = int(df[column].astype(str).str.count(";").max()) + 1
    if index >= depth:
        index = depth - 1
    name = TAXA_LEVELS[index] if index < len(TAXA_LEVELS) else f"level {index + 1}"

    taxa, confidence = split_rank(df[column], index)
    taxa = taxa.fillna(f"(no {name.lower()})")

    counts = taxa.value_counts().sort_values(ascending=False)
    conf_by_taxon = None
    if confidence.notna().any():
        conf_by_taxon = confidence.groupby(taxa).mean().reindex(counts.index)

    return RankSummary(
        rank=name,
        requested_rank=requested_name,
        counts=counts,
        confidence=conf_by_taxon,
        n_sequences=int(len(df)),
        n_unclassified=int(sum(counts[t] for t in counts.index if is_unclassified(t))),
    )


def _bar(fraction: float, width: int) -> str:
    """Draw a block bar filling ``fraction`` of ``width`` cells, 1/8-cell resolution."""
    if width <= 0 or fraction <= 0:
        return ""
    eighths = round(max(0.0, min(1.0, fraction)) * width * 8)
    full, remainder = divmod(eighths, 8)
    bar = "█" * full + (_BLOCKS[remainder] if remainder else "")
    return bar if bar.strip() else "▏"  # never render a non-zero count as blank


def _use_color(color: str | bool, stream) -> bool:
    if color in (True, False):
        return bool(color)
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def _paint(text: str, code: str, enabled: bool) -> str:
    return f"{code}{text}{_ANSI['reset']}" if enabled and text else text


def _truncate(label: str, width: int) -> str:
    return label if len(label) <= width else label[:max(1, width - 1)] + "…"


def render_report(classified: pd.DataFrame | dict,
                  *,
                  rank: str | int = "Phylum",
                  top: int = 15,
                  width: int | None = None,
                  color: str | bool = "auto",
                  column: str = "classification",
                  stream=None) -> str:
    """Build the bar-chart report as a string.

    Args:
        classified: Classification results with a lineage column.
        rank: Rank to summarize (name or 0-based index). Default "Phylum".
        top: Maximum number of taxa to chart; the remainder is folded into an
            "Other" row. Use 0 or None for no limit.
        width: Total line width. Defaults to the terminal width (max 100).
        color: True/False, or "auto" to color only when writing to a tty.
        column: Name of the lineage column.
        stream: Stream used for the "auto" color decision. Defaults to stdout.

    Returns:
        str: The rendered report, without a trailing newline.
    """
    summary = summarize_rank(classified, rank, column=column)
    painted = _use_color(color, stream or sys.stdout)

    if width is None:
        width = min(shutil.get_terminal_size((80, 24)).columns, 100)
    width = max(40, int(width))

    counts = summary.counts
    rows = list(counts.items())
    folded = []
    if top and len(rows) > top:
        folded = rows[top:]
        rows = rows[:top]

    total = summary.n_sequences
    other_total = int(sum(c for _, c in folded))
    # Bars are scaled to the largest row actually drawn, the "Other" row
    # included: at genus and below the folded tail routinely outweighs every
    # individual taxon, and clamping its bar to the full width would make it
    # look no larger than the top row.
    largest = max([int(c) for _, c in rows] + [other_total]) or 1

    other_label = f"Other ({len(folded)} taxa)" if folded else ""
    label_width = min(max((len(str(t)) for t, _ in rows), default=10), 30)
    if folded:
        label_width = max(label_width, len(other_label))
    count_width = max([len(f"{int(c)}") for _, c in rows] + [len(str(other_total))])
    # label + 2 spaces + bar + 2 spaces + count + 1 space + "100.0%"
    bar_width = max(10, width - label_width - count_width - 12)

    def row_line(label: str, count: int, code: str) -> str:
        pct = 100 * count / total if total else 0.0
        bar = _bar(count / largest if largest else 0, bar_width)
        # The bar cell is padded by hand: ANSI escapes would otherwise be
        # counted as visible characters by an f-string width spec.
        return (f"{_truncate(str(label), label_width):<{label_width}}  "
                f"{_paint(bar, code, painted)}{' ' * (bar_width - len(bar))}"
                f"  {count:>{count_width}} {pct:>5.1f}%")

    title = f"{summary.rank}-level summary"
    subtitle = (f"{total} sequence{'s' if total != 1 else ''} · "
                f"{summary.n_taxa} {'taxon' if summary.n_taxa == 1 else 'taxa'}")
    if summary.n_named_taxa != summary.n_taxa:
        # Otherwise the header count and the "Other (n taxa)" label appear to
        # contradict each other, since one excludes placeholders and one does not.
        subtitle += f" ({summary.n_named_taxa} named)"
    header = _paint(title, _ANSI['bold'], painted) + _paint(f"  {subtitle}", _ANSI['dim'], painted)
    rule = _paint("─" * width, _ANSI['dim'], painted)

    lines = [header, rule]
    if summary.truncated:
        lines.insert(1, _paint(
            f"(requested {summary.requested_rank.lower()}; classifications only reach "
            f"{summary.rank.lower()})", _ANSI['dim'], painted))

    color_i = 0
    for taxon, count in rows:
        if is_unclassified(taxon) or str(taxon).startswith("(no "):
            code = _ANSI["dim"]
        else:
            code = _BAR_COLORS[color_i % len(_BAR_COLORS)]
            color_i += 1
        lines.append(row_line(taxon, int(count), code))

    if folded:
        lines.append(row_line(other_label, other_total, _ANSI["dim"]))

    lines.append(rule)

    resolved_pct = 100 * summary.n_classified / total if total else 0.0
    footer = (f"resolved at {summary.rank.lower()}: {summary.n_classified}/{total} "
              f"({resolved_pct:.1f}%)")
    if summary.confidence is not None and summary.confidence.notna().any():
        weighted = (summary.confidence * counts).sum() / counts[summary.confidence.notna()].sum()
        footer += f" · mean confidence {weighted:.1f}"
    lines.append(_paint(footer, _ANSI["dim"], painted))

    return "\n".join(lines)


def print_report(classified: pd.DataFrame | dict,
                 *,
                 rank: str | int = "Phylum",
                 top: int = 15,
                 width: int | None = None,
                 color: str | bool = "auto",
                 column: str = "classification",
                 file=None) -> None:
    """Render the report and write it to ``file`` (stdout by default)."""
    out = file or sys.stdout
    print(render_report(classified, rank=rank, top=top, width=width,
                        color=color, column=column, stream=out), file=out)
