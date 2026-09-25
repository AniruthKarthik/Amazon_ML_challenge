"""Synthetic contract tests; no competition data is accessed."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data_contract import (
    DataContractError,
    connected_components,
    iter_source,
    load_ground_truth,
    load_source,
    main,
    summarize_training,
)


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, content):
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path

    def sources(self):
        s1 = load_source(self.write("s1.tsv", "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-a\tCafé\t\tFrance\nS1-b\tShop\tAddr\t\n"), 1)
        s2 = load_source(self.write("s2.tsv", "entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-a\tCafe\t\tFrance\nS2-b\tOther\t\t\n"), 2)
        s3 = load_source(self.write("s3.tsv", "entity_id\tbusiness_name\tbusiness_address\tcountry\nS3-a\tX\t\tIndia\n"), 3)
        return s1, s2, s3

    def test_load_preserves_raw_values_and_empty_optional_fields(self):
        s1, _, _ = self.sources()
        self.assertEqual(s1["S1-a"].business_name, "Café")
        self.assertEqual(s1["S1-a"].business_address, "")
        self.assertEqual(s1["S1-b"].country, "")

    def test_streaming_source_loader_preserves_rows(self):
        path = self.write("stream.tsv", "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-a\tCafé\t\tFrance\nS1-b\tShop\tAddr\t\n")
        records = list(iter_source(path, 1))
        self.assertEqual([record.entity_id for record in records], ["S1-a", "S1-b"])
        self.assertEqual(records[0].business_address, "")

    def test_header_order_and_quoted_tabs(self):
        path = self.write("source.tsv", 'country\tentity_id\tbusiness_address\tbusiness_name\nFrance\tS1-x\t"12\tMain"\tName\n')
        records = load_source(path, 1)
        self.assertEqual(records["S1-x"].business_address, "12\tMain")

    def test_invalid_source_rows(self):
        header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        cases = (
            "S2-a\tName\t\tUS\n",  # wrong source
            "S1-\tName\t\tUS\n",  # empty suffix
            "S1-a\t \t\tUS\n",  # missing name
            "S1-a\tName\tUS\n",  # missing field
            "S1-a\tName\t\tUS\nS1-a\tOther\t\tUS\n",  # duplicate
        )
        for row in cases:
            with self.subTest(row=row):
                with self.assertRaises(DataContractError):
                    load_source(self.write("bad.tsv", header + row), 1)

    def test_bad_headers_and_empty_file(self):
        for content in ("", "entity_id,business_name,business_address,country\n"):
            with self.subTest(content=content):
                with self.assertRaises(DataContractError):
                    load_source(self.write("bad.tsv", content), 1)

    def test_malformed_quoted_tsv_is_contract_error(self):
        path = self.write("bad.tsv", 'entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-a\t"Unclosed\t\tUS\n')
        with self.assertRaisesRegex(DataContractError, "invalid TSV"):
            load_source(path, 1)

    def test_ground_truth_and_graph_include_isolates(self):
        s1, s2, s3 = self.sources()
        path = self.write("truth.tsv", "source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a,S3-a\nS1-b\t\n")
        truth = load_ground_truth(path, s1, s2, s3)
        self.assertEqual(truth["S1-b"], frozenset())
        self.assertEqual(connected_components(s1, s2, s3, truth), (
            ("S1-a", "S2-a", "S3-a"), ("S1-b",), ("S2-b",)
        ))
        self.assertEqual(summarize_training(s1, s2, s3, truth), {
            "source_counts": {"S1": 2, "S2": 2, "S3": 1},
            "ground_truth_links": 2,
            "s1_singleton_ratio": 0.5,
            "component_count": 3,
            "component_sizes": [3, 1, 1],
        })

    def test_ground_truth_rejects_invalid_links_and_coverage(self):
        s1, s2, s3 = self.sources()
        cases = (
            "S1-a\tS2-a\n",  # missing S1-b
            "S1-x\t\nS1-a\t\nS1-b\t\n",  # unknown source
            "S1-a\t\nS1-a\t\nS1-b\t\n",  # duplicate source
            "S1-a\tS1-b\nS1-b\t\n",  # invalid target source
            "S1-a\tS2-missing\nS1-b\t\n",  # unknown target
            "S1-a\tS2-a,S2-a\nS1-b\t\n",  # duplicate in list
            "S1-a\tS2-a\nS1-b\tS2-a\n",  # duplicate across sources
            "S1-a\tS2-a,\nS1-b\t\n",  # malformed list
        )
        for rows in cases:
            with self.subTest(rows=rows):
                path = self.write("bad_truth.tsv", "source1_entity_id\tmatched_entity_ids\n" + rows)
                with self.assertRaises(DataContractError):
                    load_ground_truth(path, s1, s2, s3)

    def test_command_line_report_on_synthetic_training_files(self):
        self.write("train_source1.tsv", "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-a\tName\t\tFrance\n")
        self.write("train_source2.tsv", "entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-a\tName\t\tFrance\n")
        self.write("train_source3.tsv", "entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        self.write("train_ground_truth.tsv", "source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a\n")
        output = io.StringIO()
        with patch("sys.argv", ["data_contract.py", "--data-dir", str(self.root)]):
            with contextlib.redirect_stdout(output):
                main()
        report = json.loads(output.getvalue())
        self.assertEqual(report["source_counts"], {"S1": 1, "S2": 1, "S3": 0})
        self.assertEqual(report["component_sizes"], [2])


if __name__ == "__main__":
    unittest.main()
