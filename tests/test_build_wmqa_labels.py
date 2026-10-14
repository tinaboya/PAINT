"""Tests for src/paint/build_wmqa_labels.py (standard library only)."""
import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "paint"))
import build_wmqa_labels as b  # noqa: E402

PAIN_HEADER = ["Question_ID", "ok", "pain_related", "pain_presence", "location", "pain_type", "severity"]
DEMO_HEADER = ["Question_ID", "age", "gender", "ethnicity", "occupation"]


def _write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


class Fixture:
    """A three-item partition: items with source Question_ID 7, 8, 9."""

    def __init__(self, root):
        self.root = Path(root)
        self.data = self.root / "data" / "WorldMedQA-Pain"
        self.outputs = self.root / "model_outputs"
        _write(self.data / "Tiny.csv",
               ["Question_ID", "file_name", "Question", "Question_EN"],
               [["7", "exam.pdf", "Mulher com dor no ombro", "Woman with shoulder pain"],
                ["8", "exam.pdf", "Homem sem queixas", "Man without complaints"],
                ["9", "exam.pdf", "Crianca com febre", "Child with fever"]])
        pain = {  # (pain_related, pain_presence, location, pain_type, severity)
            "deepseek-r1-qwen-32b": [("TRUE", "TRUE", "ombro", "aguda", "moderate"), ("FALSE", "FALSE", "", "", ""), ("FALSE", "TRUE", "", "", "")],
            "gemma4-e4b":           [("TRUE", "TRUE", "ombro;braco", "aguda", "severe"), ("FALSE", "FALSE", "", "", ""), ("TRUE", "FALSE", "", "", "")],
            "qwen3-8b":             [("TRUE", "TRUE", "ombro", "", ""), ("FALSE", "FALSE", "", "", ""), ("FALSE", "FALSE", "", "", "")],
        }
        demo = {
            "deepseek-r1-qwen-32b": [("54 anos", "mulher", "", ""), ("", "homem", "", "pedreiro"), ("3 anos", "", "", "")],
            "gemma4-e4b":           [("54", "female", "", ""), ("", "male", "", "pedreiro"), ("3", "", "", "")],
            "qwen3-8b":             [("54 years", "F", "", ""), ("40", "M", "", "carpinteiro"), ("", "", "", "")],
        }
        # DeepSeek files use the source Question_ID; Gemma and Qwen pain files use the row position.
        ids = {"deepseek-r1-qwen-32b": ["7", "8", "9"], "gemma4-e4b": ["0", "1", "2"], "qwen3-8b": ["0", "1", "2"]}
        for m in b.MEMBERS:
            _write(self.outputs / m / "Tiny_pain.csv", PAIN_HEADER,
                   [[i, "TRUE", *vals] for i, vals in zip(ids[m], pain[m])])
            _write(self.outputs / m / "Tiny_demographics.csv", DEMO_HEADER,
                   [[i, *vals] for i, vals in zip(ids[m], demo[m])])

    def install(self):
        b.DATA_DIR, b.MODEL_OUTPUTS = self.data, self.outputs


class StrictRule(unittest.TestCase):
    def test_all_same_true_two_of_three_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            Fixture(tmp).install()
            rows = b.build_partition("Tiny")
        self.assertEqual([r["ensemble_pain_related"] for r in rows], ["True", "False", "abstain"])
        self.assertEqual([r["ensemble_pain_presence"] for r in rows], ["True", "False", "abstain"])

    def test_pbool_accepts_common_spellings(self):
        for v in ("TRUE", "true", "T", "1", "yes"):
            self.assertIs(b._pbool(v), True)
        for v in ("FALSE", "f", "0", "no"):
            self.assertIs(b._pbool(v), False)
        self.assertIsNone(b._pbool(""))
        self.assertIsNone(b._pbool("maybe"))


class MajorityFields(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        Fixture(self.tmp.name).install()
        self.rows = b.build_partition("Tiny")

    def tearDown(self):
        self.tmp.cleanup()

    def test_list_fields_keep_items_two_models_agree_on(self):
        self.assertEqual(self.rows[0]["ensemble_location"], "ombro")   # braco listed by one model only
        self.assertEqual(self.rows[0]["ensemble_pain_type"], "aguda")

    def test_empty_means_no_extraction_and_abstain_means_disagreement(self):
        self.assertEqual(self.rows[0]["ensemble_severity"], "abstain")  # moderate vs severe
        self.assertEqual(self.rows[1]["ensemble_severity"], "")          # nobody extracted one
        self.assertEqual(self.rows[1]["ensemble_occupation"], "pedreiro")  # 2 of 3
        self.assertEqual(self.rows[2]["ensemble_gender"], "")

    def test_cross_lingual_normalisation(self):
        self.assertEqual(self.rows[0]["ensemble_age"], "54")
        self.assertEqual(self.rows[0]["ensemble_gender"], "female")
        self.assertEqual(self.rows[1]["ensemble_gender"], "male")
        self.assertEqual(self.rows[2]["ensemble_age"], "3")

    def test_identity_and_text_columns(self):
        self.assertEqual([r["row_index"] for r in self.rows], [0, 1, 2])
        self.assertEqual(self.rows[0]["Question_ID"], "7")
        self.assertEqual(self.rows[0]["Question_EN"], "Woman with shoulder pain")


class Alignment(unittest.TestCase):
    def _mutate(self, path, fn):
        with open(path, encoding="utf-8", newline="") as fh:
            rows = list(csv.reader(fh))
        rows = [rows[0]] + fn(rows[1:])
        with open(path, "w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerows(rows)

    def test_reordered_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp); f.install()
            self._mutate(f.outputs / "gemma4-e4b" / "Tiny_pain.csv", lambda r: r[::-1])
            with self.assertRaises(SystemExit) as cm:
                b.build_partition("Tiny")
            self.assertIn("not row-aligned", str(cm.exception))

    def test_reordered_source_id_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp); f.install()
            self._mutate(f.outputs / "deepseek-r1-qwen-32b" / "Tiny_demographics.csv", lambda r: [r[1], r[0], r[2]])
            with self.assertRaises(SystemExit):
                b.build_partition("Tiny")

    def test_dropped_row_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp); f.install()
            self._mutate(f.outputs / "qwen3-8b" / "Tiny_pain.csv", lambda r: r[:-1])
            with self.assertRaises(SystemExit) as cm:
                b.build_partition("Tiny")
            self.assertIn("row counts differ", str(cm.exception))

    def test_duplicated_row_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp); f.install()
            self._mutate(f.outputs / "qwen3-8b" / "Tiny_demographics.csv", lambda r: [r[0], r[0], r[2]])
            with self.assertRaises(SystemExit):
                b.build_partition("Tiny")


class ReleasedFiles(unittest.TestCase):
    def test_check_mode_reports_zero_differences(self):
        proc = subprocess.run([sys.executable, str(ROOT / "src" / "paint" / "build_wmqa_labels.py"), "--check"],
                              capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("differing from data/WorldMedQA-Pain/: 0", proc.stdout)

    def test_check_mode_fails_on_a_changed_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp); f.install()
            rows = b.build_partition("Tiny")
            out = f.data / "Tiny.csv"
            with open(out, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=b.ID_COLUMNS + b.LABEL_COLUMNS, lineterminator="\n")
                w.writeheader(); w.writerows(rows)
            self._mutate_cell(out)
            saved = b.PARTITIONS; b.PARTITIONS = ["Tiny"]
            try:
                sys.argv = ["build_wmqa_labels.py", "--check"]
                with self.assertRaises(SystemExit) as cm:
                    b.main()
                self.assertEqual(cm.exception.code, 1)
            finally:
                b.PARTITIONS = saved

    def _mutate_cell(self, path):
        with open(path, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh)); fields = rows[0].keys()
        rows[0]["ensemble_pain_related"] = "False"
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(fields), lineterminator="\n"); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    unittest.main()
