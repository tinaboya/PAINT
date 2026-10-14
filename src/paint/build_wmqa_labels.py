"""Build the WorldMedQA-Pain release files from the per-model extractions.

Each partition CSV in ``data/WorldMedQA-Pain/`` carries the exam item text and
one ensemble column per annotated field. The three ensemble members
(Gemma-4-E4B, Qwen3-8B, DeepSeek-R1-Distill-Qwen-32B) label every item; this
script aggregates their outputs with the rules used in the paper:

* ``ensemble_pain_related``, ``ensemble_pain_presence`` -- strict all-same
  rule: the agreed value when the three models agree, otherwise ``abstain``.
* ``ensemble_location``, ``ensemble_pain_type`` -- lists merged item by item,
  an item is kept when at least two models list it (one model alone is kept
  when no other model returned a list); items are joined with ``; ``.
* ``ensemble_severity``, ``ensemble_age``, ``ensemble_gender``,
  ``ensemble_ethnicity``, ``ensemble_occupation`` -- 2-of-3 majority over the
  non-null predictions (a single prediction is kept when no model contradicts
  it).

For the majority fields an empty cell means that no model extracted the
field, and ``abstain`` means that the models disagreed without a majority;
both count as missing in the paper's coverage figures. Normalisation (age in
years, gender male/female, severity mild/moderate/severe, list items
lower-cased) is shared with ``analysis.py``.

No physician labels are used or required: the released labels are produced by
the open-weight ensemble alone. Rows are aligned by position and the alignment
is verified against the index column of every model file; ``row_index`` is the
0-based position shared with ``model_outputs/`` and the annotation sheets.

Run:
    python3 src/paint/build_wmqa_labels.py            # rewrite the CSVs
    python3 src/paint/build_wmqa_labels.py --check    # compare with the CSVs
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analysis import (  # noqa: E402
    _ensemble_demo,
    _ensemble_pain_aux,
    _normalize_demo_row,
    _normalize_pain_row,
)

ROOT = HERE.parent.parent
MODEL_OUTPUTS = ROOT / "model_outputs"
DATA_DIR = ROOT / "data" / "WorldMedQA-Pain"

MEMBERS = ["deepseek-r1-qwen-32b", "gemma4-e4b", "qwen3-8b"]
PARTITIONS = ["UK", "Spain", "Japan", "Brazil"]
STRICT_FIELDS = ["pain_related", "pain_presence"]
LIST_FIELDS = ["location", "pain_type"]
DEMO_FIELDS = ["age", "gender", "ethnicity", "occupation"]
ABSTAIN = "abstain"

ID_COLUMNS = ["row_index", "Question_ID", "file_name", "Question", "Question_EN"]
LABEL_COLUMNS = ["ensemble_" + f for f in STRICT_FIELDS + LIST_FIELDS + ["severity"] + DEMO_FIELDS]


def _read(path: Path):
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def _pbool(v):
    s = str(v).strip().lower()
    if s in ("true", "t", "1", "yes"):
        return True
    if s in ("false", "f", "0", "no"):
        return False
    return None


def _strict(pain_rows, i):
    """Strict all-same rule for the two binary pain fields."""
    out = {}
    for f in STRICT_FIELDS:
        votes = [_pbool(pain_rows[m][i].get(f)) for m in MEMBERS]
        agree = len(set(votes)) == 1 and votes[0] is not None
        out["ensemble_" + f] = str(votes[0]) if agree else ABSTAIN
    return out


def _cell(value, n_nonnull):
    """Serialise a majority-rule result: '' = no model extracted the field,
    'abstain' = models disagreed without a majority."""
    if value is None:
        return "" if n_nonnull == 0 else ABSTAIN
    return str(value)


def _majority(pain_rows, demo_rows, i):
    pain_norm = {m: _normalize_pain_row(pain_rows[m][i]) for m in MEMBERS}
    demo_norm = {m: _normalize_demo_row(demo_rows[m][i]) for m in MEMBERS}
    aux = _ensemble_pain_aux(pain_norm)
    demo = _ensemble_demo(demo_norm)
    out = {}
    for f in LIST_FIELDS:
        v = aux[f]
        if v:  # order kept items by first appearance across the members
            order = []
            for m in MEMBERS:
                for it in pain_norm[m][f] or []:
                    if it not in order:
                        order.append(it)
            v = sorted(v, key=order.index)
        out["ensemble_" + f] = "" if v is None else (ABSTAIN if v == [] else "; ".join(v))
    n = sum(1 for m in MEMBERS if pain_norm[m]["severity"] is not None)
    out["ensemble_severity"] = _cell(aux["severity"], n)
    for f in DEMO_FIELDS:
        n = sum(1 for m in MEMBERS if demo_norm[m][f] is not None)
        out["ensemble_" + f] = _cell(demo[f], n)
    return out


def check_alignment(partition, data_rows, pain_rows, demo_rows):
    """Fail unless every model-output file is in the same row order as the data file.

    Rows are joined by position, so a reordered, truncated or duplicated file
    would silently produce wrong labels. Each model file carries an index
    column under one of two schemes: the 0-based row position (``row_index``,
    or ``Question_ID`` in some files) or the WorldMedQA ``Question_ID`` of the
    item. Whichever scheme a file uses, every row must match the data file at
    the same position; row counts must be identical as well.
    """
    counts = {len(data_rows)} | {len(r) for r in pain_rows.values()} | {len(r) for r in demo_rows.values()}
    if len(counts) != 1:
        raise SystemExit(f"[{partition}] row counts differ across files: {sorted(counts)}")
    source_ids = [str(d.get("Question_ID", "")).strip() for d in data_rows]
    problems = []
    for task, files in (("pain", pain_rows), ("demographics", demo_rows)):
        for m, rows in files.items():
            col = "row_index" if "row_index" in rows[0] else "Question_ID"
            ids = [str(r.get(col, "")).strip() for r in rows]
            by_position = sum(1 for i, v in enumerate(ids) if v == str(i))
            by_source = sum(1 for i, v in enumerate(ids) if v == source_ids[i])
            if max(by_position, by_source) != len(rows):
                problems.append(f"{m}/{partition}_{task}.csv: {col} matches the row position on "
                                f"{by_position}/{len(rows)} rows and the source Question_ID on {by_source}/{len(rows)}")
    if problems:
        raise SystemExit(f"[{partition}] model outputs are not row-aligned with data/WorldMedQA-Pain/{partition}.csv:\n  "
                         + "\n  ".join(problems))


def build_partition(partition: str):
    """Return the release rows for one partition (identity + text + labels)."""
    data_rows = _read(DATA_DIR / f"{partition}.csv")
    pain_rows = {m: _read(MODEL_OUTPUTS / m / f"{partition}_pain.csv") for m in MEMBERS}
    demo_rows = {m: _read(MODEL_OUTPUTS / m / f"{partition}_demographics.csv") for m in MEMBERS}
    check_alignment(partition, data_rows, pain_rows, demo_rows)
    out = []
    for i, d in enumerate(data_rows):
        question = d.get("Question", "")
        row = {
            "row_index": i,
            "Question_ID": d.get("Question_ID", ""),
            "file_name": d.get("file_name", ""),
            "Question": question,
            # UK items are already in English; the column repeats the question.
            "Question_EN": d.get("Question_EN") or question,
        }
        row.update(_strict(pain_rows, i))
        row.update(_majority(pain_rows, demo_rows, i))
        out.append(row)
    return out


def _is_value(cell: str) -> bool:
    return cell not in ("", ABSTAIN)


def summarise(partition: str, rows) -> None:
    n = len(rows)
    resolved = [r for r in rows if r["ensemble_pain_related"] != ABSTAIN and r["ensemble_pain_presence"] != ABSTAIN]
    pres = sum(1 for r in resolved if r["ensemble_pain_presence"] == "True")
    # Table 7 "Location %": pain-presence-positive items with a non-empty location, over all items
    loc = sum(1 for r in rows if r["ensemble_pain_presence"] == "True" and _is_value(r["ensemble_location"]))
    cov = {f: sum(1 for r in resolved if _is_value(r["ensemble_" + f])) for f in DEMO_FIELDS}
    print(f"{partition:7} n={n:5}  PP-resolved={len(resolved):5} ({100*len(resolved)/n:.1f}%)  "
          f"pain-presence={pres:4} ({100*pres/n:.1f}% of n)  presence+location={loc:4} ({100*loc/n:.1f}% of n)")
    print("        demographic coverage within PP-resolved: "
          + "  ".join(f"{f}={100*cov[f]/len(resolved):.1f}%" for f in DEMO_FIELDS))
    print("        abstain per field: "
          + "  ".join(f"{c[9:]}={sum(1 for r in rows if r[c] == ABSTAIN)}" for c in LABEL_COLUMNS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="compare the recomputed columns with data/WorldMedQA-Pain/ instead of writing")
    args = ap.parse_args()
    columns = ID_COLUMNS + LABEL_COLUMNS
    total, mismatches = 0, 0
    for part in PARTITIONS:
        rows = build_partition(part)
        total += len(rows)
        path = DATA_DIR / f"{part}.csv"
        if args.check:
            current = _read(path)
            if len(current) != len(rows):
                print(f"[{part}] {path.name} has {len(current)} rows, rebuild has {len(rows)}")
                mismatches += abs(len(current) - len(rows)) * len(LABEL_COLUMNS)
            for new, old in zip(rows, current):
                for c in LABEL_COLUMNS:
                    if str(new[c]) != old.get(c, ""):
                        mismatches += 1
        else:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
        summarise(part, rows)
    print(f"total items: {total}")
    if args.check:
        print(f"label cells differing from data/WorldMedQA-Pain/: {mismatches}")
        sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
