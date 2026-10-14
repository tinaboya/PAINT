"""Inter-annotator agreement on the double-annotated overlaps.

For every partition with two clinician sheets this computes, on the overlap:

  (1) Cohen's kappa for the binary pain fields (pain_related, pain_presence),
      the paper's primary IAA metric for those fields.

  (2) Percent agreement AND Cohen's kappa for the demographic slot-filling
      fields (age, gender, occupation). Percent agreement is the primary
      metric there because kappa is undefined when the overlap holds a single
      category (chance agreement is total) and unstable on high-cardinality
      fields such as age and free-text occupation; kappa is reported alongside
      so readers can see where it is informative.

This reuses the loaders/normalisers in clinician_validation.py so the numbers
are computed exactly as in the paper.

Run:
    python3 src/paint/inter_annotator_agreement.py
"""

from __future__ import annotations

import json
import os
from importlib.machinery import SourceFileLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
cv = SourceFileLoader("cv", os.path.join(HERE, "clinician_validation.py")).load_module()

# Single source of truth: the completed annotation sheets committed to the repo
# (one filled sheet per dataset/annotator, including the second Spanish Cantemist
# Spanish annotator for Cantemist and Spain-MIR).
cv.SEARCH_DIRS = [os.path.join(ROOT, "annotation", "completed_sheets")]

OUT_PATH = os.path.join(ROOT, "outputs", "analysis", "inter_annotator_agreement.json")


def _cohen_kappa(a, b):
    """Multi-category Cohen's kappa over paired categorical labels."""
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe >= 1.0:            # all judgements collapse to one category
        return None          # kappa undefined (chance agreement is total)
    return (po - pe) / (1 - pe)


def demo_agreement(sheets, dataset, field):
    """Percent agreement AND Cohen's kappa for a demographic field on the overlap.

    Records that either annotator left unjudged (blank ``__correct`` cell) are
    excluded: a skipped field is missing data, not a "null" label, and counting
    it as one would manufacture disagreement.
    """
    A = {str(d.get("row_index")): d for d in sheets.get((dataset, "A"), [])}
    B = {str(d.get("row_index")): d for d in sheets.get((dataset, "B"), [])}
    overlap = [k for k in A if k in B
               if cv._annotated(A[k], field) and cv._annotated(B[k], field)]
    if not overlap:
        return None
    a = [cv._norm_demo(field, cv._reconstruct(A[k], field)) for k in overlap]
    b = [cv._norm_demo(field, cv._reconstruct(B[k], field)) for k in overlap]
    n = len(a)
    pct = sum(1 for x, y in zip(a, b) if x == y) / n
    return {
        "n": n,
        "n_categories": len(set(a) | set(b)),
        "pct_agree": round(pct * 100, 1),
        "cohen_kappa": (None if _cohen_kappa(a, b) is None
                        else round(_cohen_kappa(a, b), 3)),
    }


def pain_iaa(sheets, dataset, field):
    """Cohen's kappa for a binary pain field on the overlap (paper's method)."""
    A = {str(d.get("row_index")): d for d in sheets.get((dataset, "A"), [])}
    B = {str(d.get("row_index")): d for d in sheets.get((dataset, "B"), [])}
    overlap = [k for k in A if k in B]
    pairs = [(cv._pbool(cv._reconstruct(A[k], field)),
              cv._pbool(cv._reconstruct(B[k], field))) for k in overlap]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if not pairs:
        return None
    k = cv._kappa([x for x, _ in pairs], [y for _, y in pairs])
    return {"n": len(pairs),
            "cohen_kappa": (None if k is None else round(k, 3)),
            "pct_agree": round(sum(1 for x, y in pairs if x == y) / len(pairs) * 100, 1)}


def main():
    sheets, srcs = cv._discover()

    print("=== source sheet per (dataset, annotator) ===")
    for k in sorted(srcs):
        print(f"  {k}: {srcs[k]}")

    report = {"double_annotated": {}, "single_annotator": []}

    for ds in cv.BRIDGE + cv.WMQA:
        hasA, hasB = (ds, "A") in sheets, (ds, "B") in sheets
        if not (hasA and hasB):
            report["single_annotator"].append(ds)
            continue
        entry = {"pain": {}, "demographics": {}}
        fields_pain = cv.PAIN_FIELDS if ds in cv.BRIDGE else []
        for f in fields_pain:
            entry["pain"][f] = pain_iaa(sheets, ds, f)
        for f in cv.DEMO_FIELDS:
            entry["demographics"][f] = demo_agreement(sheets, ds, f)
        report["double_annotated"][ds] = entry

    print("\n=== (1) Spanish partitions ===")
    for ds in ("es_Cantemis", "es_Spain"):
        status = "DOUBLE-ANNOTATED" if ds in report["double_annotated"] else "single annotator only"
        print(f"  {ds}: {status}")

    print("\n=== (2) Demographics: percent agreement vs. Cohen's kappa ===")
    print(f"  {'dataset':12s} {'field':11s} {'n':>3s} {'#cat':>4s} {'%agree':>7s} {'kappa':>7s}")
    for ds, e in report["double_annotated"].items():
        for f, v in e["demographics"].items():
            if v is None:
                continue
            kap = "undef" if v["cohen_kappa"] is None else f"{v['cohen_kappa']:.3f}"
            print(f"  {ds:12s} {f:11s} {v['n']:>3d} {v['n_categories']:>4d} "
                  f"{v['pct_agree']:>6.1f}% {kap:>7s}")

    print("\n=== pain-field IAA (for datasets with two annotators) ===")
    for ds, e in report["double_annotated"].items():
        for f, v in e["pain"].items():
            if v is None:
                continue
            print(f"  {ds:12s} {f:13s} n={v['n']:>3d} kappa={v['cohen_kappa']} "
                  f"(%agree={v['pct_agree']})")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {OUT_PATH}")
    if report["single_annotator"]:
        print("single-annotator (no IAA possible): " + ", ".join(report["single_annotator"]))


if __name__ == "__main__":
    main()
