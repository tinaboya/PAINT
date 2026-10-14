"""Clinician validation of the ensemble silver labels.

Native-speaker physicians re-annotate samples of each corpus, judging whether
the ensemble consensus value for each field is correct
(``<field>__correct(TRUE/FALSE)``) and supplying the true value when not.
From these sheets we compute, per dataset and field, one value per unique
record (option (b)): singly-reviewed records use that annotator; doubly-
reviewed (overlap) records use the agreed label and drop disagreements
(reflected by the inter-annotator agreement instead). On those records:

  * binary pain fields: macro-F1 of the ensemble against physician annotation
    (consistent with the WorldMedQA gold benchmark, Table 1);
  * demographic fields: physician-confirmed accuracy with a Wilson 95% CI; and
  * inter-annotator agreement on the overlap: Cohen's kappa for the binary
    pain fields and percent agreement for the demographic fields.

BRIDGE-Pain validates all five fields; WorldMedQA-Pain validates demographics
only (its pain fields carry physician gold). Spanish has a single annotator,
so its splits have an accuracy but no inter-annotator agreement.

Run (needs openpyxl):
    /usr/bin/python3 src/paint/clinician_validation.py
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re

import openpyxl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(ROOT, "outputs", "analysis", "clinician_validation.json")

# Single source of truth: the completed annotation sheets committed to the repo
# (one filled sheet per dataset/annotator). The loader keeps the most-filled
# sheet per (dataset, annotator).
SEARCH_DIRS = [
    os.path.join(ROOT, "annotation", "completed_sheets"),
]

BRIDGE = ["en_MEDIQA", "en_MIMIC", "es_Cantemis", "ja_IFMIR", "pt_Brateca"]
WMQA = ["en_UK", "es_Spain", "ja_Japan", "pt_Brazil"]
PAIN_FIELDS = ["pain_related", "pain_presence"]
DEMO_FIELDS = ["age", "gender", "occupation"]
_KEY = re.compile(r"(" + "|".join(BRIDGE + WMQA) + r").*annotator([AB])", re.I)


def _load(path):
    """Return list-of-dict rows with the header row auto-detected."""
    if path.endswith(".csv"):
        # Empty CSV cells must load as None, matching openpyxl's blank cells,
        # so that a blank age/gender/occupation normalises to "null" on both
        # paths rather than "" on one and None on the other.
        rows = [[c if c != "" else None for c in r]
                for r in csv.reader(open(path, encoding="utf-8", errors="replace"))]
    else:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        rows = [list(r) for r in wb.active.iter_rows(values_only=True)]
        wb.close()
    hr = next((i for i, r in enumerate(rows[:6])
               if any(c and "__correct" in str(c) for c in r)), None)
    if hr is None:
        return []
    hdr = [str(c).strip() if c is not None else "" for c in rows[hr]]
    out = []
    for r in rows[hr + 1:]:
        if any(c is not None and str(c).strip() != "" for c in r):
            d = {hdr[i]: (r[i] if i < len(r) else None) for i in range(len(hdr))}
            # xlsx loads row_index as a float (e.g. 291.0); normalise so it
            # matches the string keys from CSV sheets when computing overlap.
            if d.get("row_index") is not None:
                s = str(d["row_index"]).strip()
                d["row_index"] = s[:-2] if s.endswith(".0") else s
            out.append(d)
    return out


def _pbool(v):
    s = str(v).strip().lower()
    if s in ("true", "t", "1", "yes"):
        return True
    if s in ("false", "f", "0", "no"):
        return False
    return None


def _filled(rows):
    cols = [f"{f}__correct(TRUE/FALSE)" for f in PAIN_FIELDS + DEMO_FIELDS]
    return sum(1 for d in rows if any(str(d.get(c) or "").strip() for c in cols))


def _discover():
    """{(dataset, 'A'/'B'): rows} keeping the most-filled sheet per slot."""
    best = {}
    for d in SEARCH_DIRS:
        for f in glob.glob(d + "/**/*.csv", recursive=True) + glob.glob(d + "/**/*.xlsx", recursive=True):
            base = os.path.basename(f)
            if "master" in base.lower() or base.startswith("~$"):
                continue
            m = _KEY.search(base)
            if not m:
                continue
            try:
                rows = _load(f)
            except Exception:
                continue
            n = _filled(rows)
            if n == 0:
                continue
            key = (m.group(1), m.group(2).upper())
            if n > best.get(key, (None, 0, None))[1]:
                best[key] = (rows, n, f)
    return {k: v[0] for k, v in best.items()}, {k: v[2] for k, v in best.items()}


def _wilson(k, n):
    if n == 0:
        return None
    p, z = k / n, 1.96
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"rate": p, "lo": centre - half, "hi": centre + half, "k": k, "n": n}


def _macro_f1(gold, pred):
    if not gold:
        return None

    def f1(cls):
        tp = sum(1 for g, p in zip(gold, pred) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gold, pred) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gold, pred) if g == cls and p != cls)
        if tp == 0:
            return 0.0
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        return 2 * prec * rec / (prec + rec)

    return (f1(0) + f1(1)) / 2


def _per_record(sheets, dataset, field):
    """One (pred, gold, correct) per unique record (option (b))."""
    cc = f"{field}__correct(TRUE/FALSE)"
    con = f"{field}__consensus"

    def info(d):
        c = _pbool(d.get(cc))
        if c is None:
            return None
        pred = _pbool(d.get(con))
        pred = False if pred is None else pred
        gold = pred if c else (not pred)
        return (int(pred), int(gold), int(c))

    A = {str(d.get("row_index")): d for d in sheets.get((dataset, "A"), [])}
    B = {str(d.get("row_index")): d for d in sheets.get((dataset, "B"), [])}
    recs, dropped = [], 0
    for k in set(A) | set(B):
        ia = info(A[k]) if k in A else None
        ib = info(B[k]) if k in B else None
        if ia and ib:
            if ia[1] != ib[1]:        # physicians disagree on the gold label
                dropped += 1
                continue
            recs.append(ia)
        elif ia or ib:
            recs.append(ia or ib)
    return recs, dropped


def _field_metric(sheets, dataset, field):
    recs, dropped = _per_record(sheets, dataset, field)
    if not recs:
        return None
    if field in PAIN_FIELDS:
        return {"metric": "macro_f1",
                "value": _macro_f1([r[1] for r in recs], [r[0] for r in recs]),
                "n": len(recs), "dropped": dropped}
    w = _wilson(sum(r[2] for r in recs), len(recs))
    return {"metric": "accuracy", **w, "dropped": dropped}


def _reconstruct(d, field):
    b = _pbool(d.get(f"{field}__correct(TRUE/FALSE)"))
    if b is None:
        return None
    return d.get(f"{field}__consensus") if b else d.get(f"{field}__true_value_if_false")


def _annotated(d, field):
    """True only if this annotator actually judged the field.

    A blank ``<field>__correct(TRUE/FALSE)`` cell means the annotator skipped
    the field, which is missing data rather than a judgement of "no value".
    Agreement is undefined on records one annotator never labelled, so such
    records are excluded from the inter-annotator statistics.
    """
    return _pbool(d.get(f"{field}__correct(TRUE/FALSE)")) is not None


def _norm_demo(field, v):
    # `v or ""` would turn a numeric 0 / 0.0 into "", silently converting a
    # recorded value of zero into "null"; test for None explicitly instead.
    s = "" if v is None else str(v).strip().lower()
    if s in ("", "null", "none", "nan"):
        return "null"
    if field == "age":
        m = re.search(r"\d+", s)
        return m.group(0) if m else "null"
    if field == "gender":
        if s.startswith("m") or "masc" in s or s == "男":
            return "male"
        if s.startswith("f") or "fem" in s or "mulher" in s or s == "女":
            return "female"
        return s
    return s


def _kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in (True, False))
    if pe == 1:
        return 1.0 if po == 1 else None
    return (po - pe) / (1 - pe)


def _iaa(sheets, dataset, fields):
    if (dataset, "A") not in sheets or (dataset, "B") not in sheets:
        return None  # single annotator
    A = {str(d.get("row_index")): d for d in sheets[(dataset, "A")]}
    B = {str(d.get("row_index")): d for d in sheets[(dataset, "B")]}
    overlap = [k for k in A if k in B]
    res = {"overlap": len(overlap)}
    for f in fields:
        if f in PAIN_FIELDS:
            pairs = [(_pbool(_reconstruct(A[k], f)), _pbool(_reconstruct(B[k], f)))
                     for k in overlap]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            res[f] = {"kappa": _kappa([x for x, _ in pairs], [y for _, y in pairs]),
                      "n": len(pairs)}
        else:
            judged = [k for k in overlap
                      if _annotated(A[k], f) and _annotated(B[k], f)]
            agree = sum(1 for k in judged
                        if _norm_demo(f, _reconstruct(A[k], f)) == _norm_demo(f, _reconstruct(B[k], f)))
            res[f] = {"pct_agree": agree / len(judged) if judged else None,
                      "n": len(judged), "skipped": len(overlap) - len(judged)}
    return res


def main():
    sheets, srcs = _discover()
    result = {"bridge": {}, "wmqa": {}}
    for ds in BRIDGE:
        result["bridge"][ds] = {
            "validation": {f: _field_metric(sheets, ds, f) for f in PAIN_FIELDS + DEMO_FIELDS},
            "iaa": _iaa(sheets, ds, PAIN_FIELDS + DEMO_FIELDS),
        }
    for ds in WMQA:
        result["wmqa"][ds] = {
            "validation": {f: _field_metric(sheets, ds, f) for f in DEMO_FIELDS},
            "iaa": _iaa(sheets, ds, DEMO_FIELDS),
        }

    print("=== source sheet picked per (dataset, annotator) ===")
    for k in sorted(srcs):
        print(f"  {k}: {srcs[k]}")

    def fmt(v):
        if v is None:
            return "--"
        if v["metric"] == "macro_f1":
            return f"F1={v['value']:.3f}(n{v['n']},drop{v['dropped']})"
        return f"acc={v['rate']*100:.1f}%(n{v['n']})"

    def show(corpus, datasets, fields):
        key = corpus.lower().split()[0]
        print(f"\n=== {corpus} (option b: agreed records) ===")
        for ds in datasets:
            r = result[key][ds]
            print(f"  {ds:13s} " + "  ".join(f"{f[:4]}:{fmt(r['validation'][f])}" for f in fields))
            iaa = r["iaa"]
            if iaa is None:
                print(f"  {'':13s} IAA: single annotator")
                continue
            parts = []
            for f in fields:
                c = iaa[f]
                if "kappa" in c:
                    parts.append(f"{f[:4]} k={c['kappa']:.2f}" if c["kappa"] is not None else f"{f[:4]} k=NA")
                else:
                    parts.append(f"{f[:4]}={c['pct_agree']*100:.0f}%")
            print(f"  {'':13s} IAA(ov={iaa['overlap']}): " + "  ".join(parts))

    show("BRIDGE", BRIDGE, PAIN_FIELDS + DEMO_FIELDS)
    show("WMQA demographics", WMQA, DEMO_FIELDS)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
