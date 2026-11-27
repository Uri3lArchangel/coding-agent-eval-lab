import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from retrieval_lab import Index, METHODS, measures, ranking, read_fixture, rrf, tokens

HERE = Path(__file__).resolve().parent


class RetrievalTests(unittest.TestCase):
    def test_identifier_components(self):
        self.assertEqual(list(tokens("pkg/HTTPServer2D_test.py")),
                         ["pkg", "httpserver2d", "http", "server2", "d", "test"])

    def test_ties_independent_of_input_order(self):
        self.assertEqual(ranking({"c": 1, "a": 1, "b": 0}), ["a", "c"])
        self.assertEqual(rrf(["b", "a"], ["a", "b"]), ["a", "b"])

    def test_no_evidence_does_not_fill_to_k(self):
        index = Index({"b.py": "banana", "a.py": "apple"}, METHODS)
        for method in METHODS:
            self.assertEqual(index.search(method, "zebrafish"), [])
            self.assertEqual(index.search(method, "apricot"), [])

    def test_recall_differs_from_hit_rate_and_mrr_is_untruncated(self):
        self.assertEqual(measures(["a", "b", "c"], ["a", "c"], 2), (0.5, 1.0))
        self.assertEqual(measures(["a", "b", "c"], ["c"], 2), (0, 1/3))
        self.assertEqual(measures([], ["b"], 2), (0, 0))

    def test_data_validation(self):
        base = dict(name="sample", documents=[dict(path="a.py", text="alpha")],
                    queries=[dict(id="1", group="paraphrase", text="alpha", relevant=["a.py"])])
        cases = [
            lambda x: x["documents"].append(x["documents"][0]),
            lambda x: x["documents"][0].update(path="../escape.py"),
            lambda x: x["queries"][0].update(relevant=["not-indexed.py"]),
            lambda x: x["queries"][0].update(relevant=["a.py", "a.py"]),
            lambda x: x["queries"][0].update(group="undeclared"),
            lambda x: x["queries"].append(x["queries"][0]),
        ]
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "case.json"
            for mutate in cases:
                with self.subTest(mutate=cases.index(mutate)):
                    data = json.loads(json.dumps(base))
                    mutate(data)
                    path.write_text(json.dumps(data))
                    with self.assertRaises(ValueError):
                        read_fixture(path)
            path.write_text('{"name":"dup","name":"ignored"}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                read_fixture(path)

    def test_cli_rejects_bad_arguments(self):
        result = subprocess.run([sys.executable, str(HERE / "retrieval_lab.py"), "--repeats", "0"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--repeats", result.stderr)

    def test_cli_report_and_cross_seed_determinism(self):
        def run(seed):
            output = subprocess.check_output([sys.executable, str(HERE / "retrieval_lab.py"),
                                             "--repeats", "2"],
                                             env={**os.environ, "PYTHONHASHSEED": str(seed)})
            return json.loads(output)
        a, b = run(1), run(94)
        self.assertEqual(a["fixture_sha256"], b["fixture_sha256"])
        self.assertEqual(a["queries"], b["queries"])
        self.assertEqual(a["summary"], b["summary"])
        self.assertEqual(a["files"], 24)
        self.assertEqual(a["summary"]["all"]["queries"], 16)
        self.assertEqual(a["summary"]["all"]["methods"]["dense"]["recall@5"], 0.8125)
        self.assertEqual(a["summary"]["all"]["hybrid_vs_dense_rr"],
                         dict(wins=0, ties=15, losses=1))
        for p in a["profile"].values():
            self.assertEqual(p["samples"], 32)
            self.assertGreaterEqual(p["index_peak_python_bytes"], p["index_retained_python_bytes"])
            self.assertGreater(p["warm_p95_ms"], 0)


if __name__ == "__main__":
    unittest.main()
