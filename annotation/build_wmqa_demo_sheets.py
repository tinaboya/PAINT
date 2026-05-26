#!/usr/bin/env python3
"""Build human-annotation sheets for WorldMedQA *demographic* extraction.

WorldMedQA ships gold pain labels but NO demographic labels, yet the paper
reports demographic results, so we validate them by hand. Rows and consensus
values come directly from the paper pipeline
(`paint.analysis.load_worldmedqa`): age / gender / occupation are the
3-model union-with-majority ensemble (`_ensemble_demo`) after cross-lingual
canonicalisation. Pain is NOT re-annotated (physician gold already exists).

Lighter than BRIDGE (vignettes state age/gender explicitly, near-ceiling):
50 sampled per split, annotator A (rows 1-30), B (rows 21-50), 10 overlap.
"""
import csv
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from paint.analysis import load_worldmedqa, WMQA_FILES  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sheets_wmqa")

SEED = 42
N_SAMPLE = 50
N_PER_ANNOTATOR = 30
OVERLAP = 10

DEMO_FIELDS = ["age", "gender", "occupation"]

# country -> (output slug, language)
SPLITS = [
    ("Brazil", "pt_Brazil", "pt"),
    ("Japan",  "ja_Japan",  "ja"),
    ("Spain",  "es_Spain",  "es"),
    ("UK",     "en_UK",     "en"),
]


def _read_questions(country):
    path = WMQA_FILES[country]["gt"]
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _fmt(v):
    if v is None or v == "":
        return ""
    return str(v)


def build_split(country, slug, lang):
    records = load_worldmedqa(country)
    src = _read_questions(country)
    rows = []
    for r in records:
        i = r["row_idx"]
        q = src[i] if i < len(src) else {}
        rec = {
            "row_index": i,
            "question_id": q.get("Question_ID") or q.get("﻿Question_ID"),
            "dataset": slug,
            "language": lang,
            "input_text": q.get("Question") or q.get("﻿Question") or "",
        }
        for fld in DEMO_FIELDS:
            rec[f"{fld}__consensus"] = _fmt(r.get(fld))
        rows.append(rec)
    rng = random.Random(f"{SEED}-{slug}")
    return rng.sample(rows, min(N_SAMPLE, len(rows)))


def annotator_columns():
    cols = ["row_index", "question_id", "dataset", "language", "input_text"]
    for fld in DEMO_FIELDS:
        cols += [f"{fld}__consensus", f"{fld}__correct(TRUE/FALSE)", f"{fld}__true_value_if_false"]
    return cols


def write_annotator_csv(path, sample_rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(annotator_columns())
        for r in sample_rows:
            row = [r["row_index"], r["question_id"], r["dataset"], r["language"], r["input_text"]]
            for fld in DEMO_FIELDS:
                row += [r.get(f"{fld}__consensus", ""), "", ""]
            w.writerow(row)


def write_master_csv(path, sample_rows):
    cols = ["row_index", "question_id", "dataset", "language", "input_text"]
    cols += [f"{fld}__consensus" for fld in DEMO_FIELDS]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sample_rows:
            row = [r["row_index"], r["question_id"], r["dataset"], r["language"], r["input_text"]]
            row += [r.get(f"{fld}__consensus", "") for fld in DEMO_FIELDS]
            w.writerow(row)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("split       sampled  A    B    overlap   age_cov gender_cov occ_cov")
    for country, slug, lang in SPLITS:
        sample = build_split(country, slug, lang)
        a = sample[:N_PER_ANNOTATOR]
        b = sample[N_PER_ANNOTATOR - OVERLAP:]
        write_annotator_csv(os.path.join(OUT, f"{slug}_annotatorA.csv"), a)
        write_annotator_csv(os.path.join(OUT, f"{slug}_annotatorB.csv"), b)
        write_master_csv(os.path.join(OUT, f"{slug}_master.csv"), sample)

        def cov(fld):
            return sum(1 for r in sample if r[f"{fld}__consensus"])
        print(f"{slug:11} {len(sample):5}  {len(a):3}  {len(b):3}  {OVERLAP:5}   "
              f"{cov('age'):5}   {cov('gender'):6}   {cov('occupation'):5}")
    print(f"\nWrote sheets to {OUT}")


if __name__ == "__main__":
    main()
