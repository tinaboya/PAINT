#!/usr/bin/env python3
"""Build human-annotation sheets for the 5 BRIDGE datasets.

Rows and consensus values are taken DIRECTLY from the paper pipeline
(`paint.bridge_analysis.load_bridge_task`), so every sheet row is exactly
the gold record the paper reports:

  - pain: pain_related / pain_presence — strict all-same across the 3 models
          (Table 1 substrate). We keep only rows where BOTH are resolved.
  - demographics: age / gender / occupation — the 3-model union-with-majority
          ensemble (`_ensemble_demo`), after cross-lingual canonicalisation
          (so "masculino" == "Male", "41 anos" == 41, ...).

Input text + original id are joined by row position from BRIDGE_result/.

Sampling: 150 per dataset (fixed seed), split into annotator A (rows 1-100)
and B (rows 51-150) with a 50-row overlap for inter-annotator agreement.
Annotators fill `__correct(TRUE/FALSE)` and `__true_value_if_false`.
"""
import csv
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from paint.bridge_analysis import load_bridge_task  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sheets")
BRIDGE_RESULT = os.path.join(ROOT, "BRIDGE_result")
INPUT_MODEL = "DeepSeek-R1-Distill-Qwen-32B"

SEED = 42
N_SAMPLE = 150
N_PER_ANNOTATOR = 100
OVERLAP = 50

PAIN_FIELDS = ["pain_related", "pain_presence"]
DEMO_FIELDS = ["age", "gender", "occupation"]
ALL_FIELDS = PAIN_FIELDS + DEMO_FIELDS

# task -> (output slug, language, BRIDGE_result pain folder for input text)
DATASETS = [
    ("MEDIQA",    "en_MEDIQA",   "en", "pain.en.MEDIQA 2023-chat-A"),
    ("MIMIC",     "en_MIMIC",    "en", "pain.en.MIMIC-III Outcome.Mortality"),
    ("Cantemist", "es_Cantemis", "es", "pain.es.Cantemis-NER"),
    ("IFMIR",     "ja_IFMIR",    "ja", "pain.ja.IFMIR-Incident type"),
    ("Brateca",   "pt_Brateca",  "pt", "pain.pt.Brateca-Mortality"),
]


def _load_input_rows(folder):
    paths = glob.glob(os.path.join(BRIDGE_RESULT, folder, INPUT_MODEL, "*.json"))
    if not paths:
        raise SystemExit(f"No input json under {folder}/{INPUT_MODEL}")
    with open(paths[0]) as f:
        return json.load(f)


def _fmt(v):
    """Display an ensemble value; blank means the ensemble extracted nothing."""
    if v is None or v == "":
        return ""
    if v is True:
        return "True"
    if v is False:
        return "False"
    return str(v)


def build_dataset(task, slug, lang, folder):
    records = load_bridge_task(task)
    inputs = _load_input_rows(folder)

    rows = []
    for r in records:
        # Table 1 consensus substrate: both pain fields resolved.
        if r["pain_presence"] is None or r["pain_related"] is None:
            continue
        i = r["row_idx"]
        src = inputs[i] if i < len(inputs) else {}
        rec = {
            "row_index": i,
            "orig_id": src.get("id"),
            "dataset": slug,
            "language": lang,
            "input_text": src.get("input", ""),
        }
        for fld in ALL_FIELDS:
            rec[f"{fld}__consensus"] = _fmt(r.get(fld))
        rows.append(rec)

    rng = random.Random(f"{SEED}-{slug}")
    sample = rng.sample(rows, min(N_SAMPLE, len(rows)))
    return sample, len(records), len(rows)


def annotator_columns():
    cols = ["row_index", "orig_id", "dataset", "language", "input_text"]
    for fld in ALL_FIELDS:
        cols += [f"{fld}__consensus", f"{fld}__correct(TRUE/FALSE)", f"{fld}__true_value_if_false"]
    return cols


def write_annotator_csv(path, sample_rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(annotator_columns())
        for r in sample_rows:
            row = [r["row_index"], r["orig_id"], r["dataset"], r["language"], r["input_text"]]
            for fld in ALL_FIELDS:
                row += [r.get(f"{fld}__consensus", ""), "", ""]
            w.writerow(row)


def write_master_csv(path, sample_rows):
    cols = ["row_index", "orig_id", "dataset", "language", "input_text"]
    cols += [f"{fld}__consensus" for fld in ALL_FIELDS]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sample_rows:
            row = [r["row_index"], r["orig_id"], r["dataset"], r["language"], r["input_text"]]
            row += [r.get(f"{fld}__consensus", "") for fld in ALL_FIELDS]
            w.writerow(row)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("dataset            total  pain-agree  sampled  A    B    overlap")
    for task, slug, lang, folder in DATASETS:
        sample, total, eligible = build_dataset(task, slug, lang, folder)
        a = sample[:N_PER_ANNOTATOR]
        b = sample[N_PER_ANNOTATOR - OVERLAP:]
        write_annotator_csv(os.path.join(OUT, f"{slug}_annotatorA.csv"), a)
        write_annotator_csv(os.path.join(OUT, f"{slug}_annotatorB.csv"), b)
        write_master_csv(os.path.join(OUT, f"{slug}_master.csv"), sample)
        print(f"{slug:18} {total:5}  {eligible:10}  {len(sample):7}  {len(a):3}  {len(b):3}  {OVERLAP:5}")
    print(f"\nWrote sheets to {OUT}")


if __name__ == "__main__":
    main()
