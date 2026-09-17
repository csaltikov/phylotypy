import io
import unittest

import pandas as pd

from phylotypy import terminal_report


def make_results(rows):
    """rows: list of (id, lineage) tuples."""
    return pd.DataFrame(rows, columns=["id", "classification"])


LINEAGES = [
    ("ASV1", "Bacteria(100);Pseudomonadota(100);Gammaproteobacteria(100);Chromatiales(98)"),
    ("ASV2", "Bacteria(100);Pseudomonadota(96);Alphaproteobacteria(90);Rhizobiales(88)"),
    ("ASV3", "Bacteria(100);Bacillota(92);Bacilli(90);Bacillales(85)"),
    ("ASV4", "Bacteria(100);Bacteria_unclassified(60);Bacteria_unclassified(60);"
             "Bacteria_unclassified(60)"),
]


class TestResolveRank(unittest.TestCase):
    def test_accepts_rank_names_case_insensitively(self):
        self.assertEqual(terminal_report.resolve_rank("PHYLUM"), (1, "Phylum"))
        self.assertEqual(terminal_report.resolve_rank(" genus "), (5, "Genus"))

    def test_domain_is_an_alias_for_kingdom(self):
        self.assertEqual(terminal_report.resolve_rank("domain"), (0, "Kingdom"))

    def test_accepts_integer_indices(self):
        self.assertEqual(terminal_report.resolve_rank(2), (2, "Class"))
        self.assertEqual(terminal_report.resolve_rank("2"), (2, "Class"))

    def test_rejects_unknown_ranks(self):
        with self.assertRaises(terminal_report.ReportError):
            terminal_report.resolve_rank("subphylum")

    def test_rejects_out_of_range_indices(self):
        with self.assertRaises(terminal_report.ReportError):
            terminal_report.resolve_rank(9)


class TestSplitRank(unittest.TestCase):
    def test_strips_confidence_annotations(self):
        taxa, conf = terminal_report.split_rank(make_results(LINEAGES)["classification"], 1)
        self.assertEqual(list(taxa), ["Pseudomonadota", "Pseudomonadota", "Bacillota",
                                      "Bacteria_unclassified"])
        self.assertEqual(list(conf), [100.0, 96.0, 92.0, 60.0])

    def test_rows_shallower_than_the_rank_are_missing(self):
        shallow = make_results([("ASV1", "Bacteria(100);Bacillota(92)")])
        taxa, conf = terminal_report.split_rank(shallow["classification"], 4)
        self.assertTrue(taxa.isna().all())
        self.assertTrue(conf.isna().all())

    def test_handles_lineages_without_confidence_values(self):
        plain = make_results([("ASV1", "Bacteria;Bacillota;Bacilli")])
        taxa, conf = terminal_report.split_rank(plain["classification"], 1)
        self.assertEqual(list(taxa), ["Bacillota"])
        self.assertTrue(conf.isna().all())


class TestSummarizeRank(unittest.TestCase):
    def setUp(self):
        self.results = make_results(LINEAGES)

    def test_counts_sequences_per_taxon_descending(self):
        summary = terminal_report.summarize_rank(self.results, "phylum")
        self.assertEqual(summary.counts.iloc[0], 2)
        self.assertEqual(summary.counts.index[0], "Pseudomonadota")
        self.assertEqual(summary.n_sequences, 4)

    def test_unclassified_placeholders_are_counted_separately(self):
        summary = terminal_report.summarize_rank(self.results, "phylum")
        self.assertEqual(summary.n_unclassified, 1)
        self.assertEqual(summary.n_classified, 3)
        self.assertEqual(summary.n_taxa, 3)        # placeholder included
        self.assertEqual(summary.n_named_taxa, 2)  # placeholder excluded

    def test_mean_confidence_is_per_taxon(self):
        summary = terminal_report.summarize_rank(self.results, "phylum")
        self.assertAlmostEqual(summary.confidence["Pseudomonadota"], 98.0)
        self.assertAlmostEqual(summary.confidence["Bacillota"], 92.0)

    def test_confidence_is_none_when_lineages_carry_no_annotations(self):
        plain = make_results([("ASV1", "Bacteria;Bacillota"), ("ASV2", "Bacteria;Bacillota")])
        summary = terminal_report.summarize_rank(plain, "phylum")
        self.assertIsNone(summary.confidence)

    def test_falls_back_to_the_deepest_available_rank(self):
        summary = terminal_report.summarize_rank(self.results, "species")
        self.assertEqual(summary.rank, "Order")
        self.assertEqual(summary.requested_rank, "Species")
        self.assertTrue(summary.truncated)

    def test_accepts_a_dict_of_results(self):
        summary = terminal_report.summarize_rank(self.results.to_dict("list"), "phylum")
        self.assertEqual(summary.n_sequences, 4)

    def test_rejects_results_without_a_classification_column(self):
        with self.assertRaises(terminal_report.ReportError):
            terminal_report.summarize_rank(pd.DataFrame({"id": ["ASV1"]}), "phylum")

    def test_rejects_empty_results(self):
        with self.assertRaises(terminal_report.ReportError):
            terminal_report.summarize_rank(make_results([]), "phylum")


class TestBar(unittest.TestCase):
    def test_full_fraction_fills_the_width(self):
        self.assertEqual(terminal_report._bar(1.0, 10), "█" * 10)

    def test_zero_count_renders_nothing(self):
        self.assertEqual(terminal_report._bar(0.0, 10), "")

    def test_tiny_nonzero_fractions_still_show_a_sliver(self):
        self.assertEqual(terminal_report._bar(0.0001, 10), "▏")

    def test_partial_fractions_use_eighth_blocks(self):
        self.assertEqual(terminal_report._bar(0.5, 4), "██")
        self.assertEqual(terminal_report._bar(0.25, 2), "▌")


class TestRenderReport(unittest.TestCase):
    def setUp(self):
        self.results = make_results(LINEAGES)

    def render(self, **kwargs):
        kwargs.setdefault("color", False)
        kwargs.setdefault("width", 70)
        return terminal_report.render_report(self.results, **kwargs)

    def test_defaults_to_the_phylum_rank(self):
        self.assertIn("Phylum-level summary", self.render())

    def test_lists_taxa_with_counts_and_percentages(self):
        report = self.render()
        self.assertRegex(report, r"Pseudomonadota\s+█+\s+2\s+50\.0%")
        self.assertRegex(report, r"Bacillota\s+█+\s+1\s+25\.0%")

    def test_footer_reports_resolution_and_confidence(self):
        report = self.render()
        self.assertIn("resolved at phylum: 3/4 (75.0%)", report)
        self.assertIn("mean confidence", report)

    def test_top_folds_the_tail_into_an_other_row(self):
        report = self.render(top=1)
        self.assertIn("Other (2 taxa)", report)

    def test_a_dominant_other_row_is_not_clamped_to_the_top_rows_bar(self):
        # At deep ranks the folded tail routinely outweighs every single taxon.
        # Its bar must be the longest one, not merely tied with the top row.
        tail = [("ASV1", "Bacteria(100);Pseudomonadota(100)")]
        tail += [(f"ASV{i}", f"Bacteria(100);Phylum{i}(100)") for i in range(2, 40)]
        report = terminal_report.render_report(make_results(tail), rank="phylum",
                                               top=1, color=False, width=70)
        lines = [ln for ln in report.splitlines() if "█" in ln]
        top_bar = lines[0].count("█")
        other_bar = next(ln for ln in lines if ln.startswith("Other")).count("█")
        self.assertGreater(other_bar, top_bar)

    def test_header_notes_named_taxa_when_placeholders_are_present(self):
        self.assertIn("3 taxa (2 named)", self.render())

    def test_header_omits_the_named_count_when_nothing_is_unclassified(self):
        clean = make_results([("ASV1", "Bacteria(100);Bacillota(92)"),
                              ("ASV2", "Bacteria(100);Bacteroidota(95)")])
        report = terminal_report.render_report(clean, color=False, width=60)
        self.assertIn("2 taxa", report)
        self.assertNotIn("named", report)

    def test_header_count_and_other_label_add_up(self):
        report = self.render(top=1)
        self.assertIn("3 taxa (2 named)", report)
        self.assertIn("Other (2 taxa)", report)  # 1 charted + 2 folded == 3

    def test_no_color_leaves_no_escape_sequences(self):
        self.assertNotIn("\033", self.render())

    def test_color_adds_escape_sequences(self):
        self.assertIn("\033", self.render(color=True))

    def test_lines_fit_within_the_requested_width(self):
        for line in self.render(width=60).splitlines():
            self.assertLessEqual(len(line), 60, msg=line)

    def test_notes_when_the_requested_rank_is_deeper_than_the_data(self):
        self.assertIn("classifications only reach order", self.render(rank="species"))

    def test_singular_taxon_count(self):
        single = make_results([("ASV1", "Bacteria(100);Bacillota(92)")])
        report = terminal_report.render_report(single, color=False, width=60)
        self.assertIn("1 taxon", report)


class TestPrintReport(unittest.TestCase):
    def test_writes_the_report_to_the_given_stream(self):
        buf = io.StringIO()
        terminal_report.print_report(make_results(LINEAGES), color=False, width=70, file=buf)
        out = buf.getvalue()
        self.assertIn("Phylum-level summary", out)
        self.assertTrue(out.endswith("\n"))

    def test_a_non_tty_stream_is_not_colored(self):
        buf = io.StringIO()
        terminal_report.print_report(make_results(LINEAGES), width=70, file=buf)
        self.assertNotIn("\033", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
