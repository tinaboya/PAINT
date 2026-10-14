"""Why the reported inter-annotator agreement is informative.

Perfect or near-perfect kappa is often suspected of reflecting a degenerate
label distribution rather than genuine agreement. This script produces the
evidence to judge that directly:

  (1) Chance agreement p_e per dataset x field. On many-valued fields p_e is
      tiny, so perfect observed agreement cannot be attributed to a skewed
      label distribution.
  (2) A corpus-wide inventory of every observed disagreement, showing that
      the protocol registers disagreement wherever the source text is
      ambiguous, and that kappa = 1.00 is reached on several partitions, not
      only one.
  (3) Marginal label distributions for the Cantemist partition.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from importlib.machinery import SourceFileLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
cv = SourceFileLoader("cv", os.path.join(HERE, "clinician_validation.py")).load_module()
cv.SEARCH_DIRS = [os.path.join(ROOT, "annotation", "completed_sheets")]

OUT_PATH = os.path.join(ROOT, "outputs", "analysis", "kappa_transparency.json")
DEMO_FIELDS = ("age", "gender", "occupation")
PAIN_FIELDS = ("pain_related", "pain_presence")


def _pe(a, b):
    """Expected agreement by chance from the two annotators' marginals."""
    n = len(a)
    ca, cb = Counter(a), Counter(b)
    return sum((ca[c] / n) * (cb.get(c, 0) / n) for c in set(a) | set(b))


def _paired(sheets, dataset, field, pain):
    """Aligned (row_index, label_A, label_B) triples on the double-annotated overlap."""
    A = {str(d.get("row_index")): d for d in sheets.get((dataset, "A"), [])}
    B = {str(d.get("row_index")): d for d in sheets.get((dataset, "B"), [])}
    out = []
    for k in A:
        if k not in B:
            continue
        if pain:
            x = cv._pbool(cv._reconstruct(A[k], field))
            y = cv._pbool(cv._reconstruct(B[k], field))
        else:
            if not (cv._annotated(A[k], field) and cv._annotated(B[k], field)):
                continue
            x = cv._norm_demo(field, cv._reconstruct(A[k], field))
            y = cv._norm_demo(field, cv._reconstruct(B[k], field))
        if x is None or y is None:
            continue
        out.append((k, x, y))
    return out


def main():
    sheets, _ = cv._discover()
    datasets = sorted({d for d, _ in sheets})
    rows, disagreements, report = [], [], {}

    for ds in datasets:
        for field, pain in [(f, True) for f in PAIN_FIELDS] + \
                           [(f, False) for f in DEMO_FIELDS]:
            trip = _paired(sheets, ds, field, pain)
            if not trip:
                continue
            a = [x for _, x, _ in trip]
            b = [y for _, _, y in trip]
            n = len(trip)
            po = sum(1 for x, y in zip(a, b) if x == y) / n
            pe = _pe(a, b)
            kap = None if pe >= 1.0 else (po - pe) / (1 - pe)
            rows.append({"dataset": ds, "field": field, "n": n,
                         "n_categories": len(set(a) | set(b)),
                         "pct_agree": round(po * 100, 1),
                         "p_e": round(pe, 4),
                         "kappa": None if kap is None else round(kap, 3)})
            for k, x, y in trip:
                if x != y:
                    disagreements.append({"dataset": ds, "field": field,
                                          "row_index": k, "A": str(x), "B": str(y)})

    print("=== (1) Observed vs. chance agreement ===")
    print(f"  {'dataset':12s} {'field':14s} {'n':>3s} {'#cat':>5s} "
          f"{'%agree':>7s} {'p_e':>7s} {'kappa':>7s}")
    for r in rows:
        kap = "undef" if r["kappa"] is None else f"{r['kappa']:.3f}"
        print(f"  {r['dataset']:12s} {r['field']:14s} {r['n']:>3d} "
              f"{r['n_categories']:>5d} {r['pct_agree']:>6.1f}% "
              f"{r['p_e']:>7.4f} {kap:>7s}")

    print(f"\n=== (2) Every observed disagreement (n={len(disagreements)}) ===")
    for d in disagreements:
        print(f"  {d['dataset']:12s} {d['field']:14s} row={d['row_index']:>5s} "
              f"A={d['A'][:28]:30s} B={d['B'][:28]}")

    print("\n=== (3) Cantemist marginal label distributions ===")
    for field, pain in [(f, True) for f in PAIN_FIELDS] + \
                       [(f, False) for f in DEMO_FIELDS]:
        trip = _paired(sheets, "es_Cantemis", field, pain)
        if not trip:
            continue
        c = Counter(x for _, x, _ in trip)
        top = ", ".join(f"{k}:{v}" for k, v in c.most_common(6))
        print(f"  {field:14s} n={len(trip):>3d} distinct={len(c):>3d}  {top}")
        report[field] = {"n": len(trip), "distinct": len(c), "marginals": dict(c)}

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    json.dump({"agreement": rows, "disagreements": disagreements,
               "cantemist_marginals": report},
              open(OUT_PATH, "w"), indent=2, ensure_ascii=False, default=str)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
