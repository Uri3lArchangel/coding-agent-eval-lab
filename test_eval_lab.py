"""Black-box CLI verification with synthetic recorded outcomes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from eval_lab import equal

ROOT = Path(__file__).resolve().parent


def run(baseline="baseline.json", variant="candidate.json", *flags):
    return subprocess.run(
        [sys.executable, "eval_lab.py", str(baseline), str(variant), *flags],
        cwd=ROOT, capture_output=True, text=True, timeout=10,
    )


class ReplayTests(unittest.TestCase):
    def test_candidate_executes_contracts(self):
        outcome = run()
        self.assertEqual(outcome.returncode, 0, outcome.stderr)
        report = json.loads(outcome.stdout)
        self.assertEqual(report["overall"], {
            "tasks": 4, "baseline_passed": 2, "variant_passed": 4, "delta_tasks": 2,
            "regressions": [], "improvements": ["chunk-tail", "run-length"],
        })
        self.assertEqual(sum(task["variant"]["cases_executed"] for task in report["tasks"]), 16)
        chunk = next(t for t in report["tasks"] if t["task_id"] == "chunk-tail")
        self.assertEqual(chunk["baseline"]["failed_cases"], ["partial", "short"])
        for key in ("suite_sha256", "artifact_registry_sha256", "prompt_library_sha256"):
            self.assertRegex(report[key], "^[0-9a-f]{64}$")
        self.assertEqual(report["runs"]["baseline"]["source"], "scripted")
        self.assertEqual(report["runs"]["baseline"]["prompt_id"], "minimal-v1")
        self.assertEqual(report["runs"]["variant"]["prompt_id"], "contract-first-v2")

    def test_positive_net_cannot_hide_regression(self):
        outcome = run("baseline.json", "regression.json")
        self.assertEqual(outcome.returncode, 1)
        report = json.loads(outcome.stdout)
        self.assertEqual(report["overall"]["delta_tasks"], 1)
        self.assertEqual(report["categories"]["sequence"]["regressions"], ["stable-unique"])
        self.assertEqual(len(report["gate"]["reasons"]), 2)

    def test_category_gate_with_relaxed_overall(self):
        outcome = run("baseline.json", "regression.json", "--max-regressions", "1")
        self.assertEqual(outcome.returncode, 1)
        self.assertEqual(json.loads(outcome.stdout)["gate"]["reasons"], [
            "category regression budget exceeded: sequence",
        ])
        allowed = run("baseline.json", "regression.json", "--max-regressions", "1",
                      "--max-category-regressions", "1")
        self.assertEqual(allowed.returncode, 0)

    def test_repeatable_bytes(self):
        self.assertEqual(run().stdout, run().stdout)

    def test_typed_contracts_distinguish_boolean_and_integer(self):
        self.assertFalse(equal([1], [True]))

    def variant_with(self, mutate):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "variant.json"
            value = json.loads((ROOT / "candidate.json").read_text())
            mutate(value)
            path.write_text(json.dumps(value))
            return run("baseline.json", path)

    def test_bad_inputs_fail_without_report(self):
        for mutate, message in (
            (lambda x: x["outcomes"].pop(), "expected 4 paired outcomes"),
            (lambda x: x["outcomes"][1].update(task_id="stable-unique"), "repeated task ID"),
            (lambda x: x["outcomes"][0].update(artifact="not-registered"), "unknown trusted artifact"),
            (lambda x: x["outcomes"][0].update(output_tokens=True), "expected an integer"),
            (lambda x: x["outcomes"][0].update(trace=["other"]), "invalid trace events"),
            (lambda x: x.update(schema_version=True), "expected an integer"),
            (lambda x: x.update(source=[]), "invalid source"),
            (lambda x: x.update(prompt_id="unrecorded-v3"), "unknown prompt ID"),
            (lambda x: x.update(new_field=True), "expected exactly"),
        ):
            with self.subTest(message=message):
                outcome = self.variant_with(mutate)
                self.assertEqual(outcome.returncode, 2)
                self.assertEqual(outcome.stdout, "")
                self.assertIn(message, outcome.stderr)

    def test_bad_gate_budget(self):
        outcome = run("baseline.json", "candidate.json", "--max-regressions", "-1")
        self.assertEqual(outcome.returncode, 2)

    def test_abstention_runs_no_contracts(self):
        outcome = self.variant_with(lambda x: x["outcomes"][0].update(artifact=None, trace=[]))
        self.assertEqual(outcome.returncode, 1)
        row = next(row for row in json.loads(outcome.stdout)["tasks"]
                   if row["task_id"] == "stable-unique")
        self.assertEqual(row["variant"]["status"], "abstained")
        self.assertEqual(row["variant"]["cases_executed"], 0)

    def test_rejects_duplicate_keys_and_nonfinite_literals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "variant.json"
            for payload, message in (
                ('{"label":"x","label":"y"}', "duplicate JSON key"),
                ('{"label":NaN}', "invalid number"),
            ):
                path.write_text(payload)
                outcome = run("baseline.json", path)
                self.assertEqual(outcome.returncode, 2)
                self.assertIn(message, outcome.stderr)


if __name__ == "__main__":
    unittest.main()
