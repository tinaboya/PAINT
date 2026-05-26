"""BRIDGE-Pain analysis using per-record ensemble outputs.

For each BRIDGE task (MIMIC, Cantemist, IFMIR, Brateca, MEDIQA):
  - Load 3-model pain CSVs (model_outputs/<model>/BRIDGE_<task>_pain.csv)
  - Load 3-model demo CSVs (model_outputs/<model>/BRIDGE_<task>_demographics.csv)
  - Apply ensemble:
      * pain_related, pain_presence: all-same agreement (high-precision binary)
      * demographic fields: union with majority tie-break (per analysis.py)
  - Compute:
      * Pain density (ensemble pain-positive rate)
      * Demographic coverage (per field)
      * Pain rate stratified by gender / age_group / ethnicity
      * Cross-corpus comparison vs WMQA partitions

Output: outputs/analysis/bridge_pain_analysis.json
Run: python3 src/paint/bridge_analysis.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))
from analysis import (  # type: ignore
    parse_age, age_group, normalize_gender, normalize_ethnicity,
    parse_bool, chi_square, DEMOGRAPHIC_MODEL_DIRS,
    _read_csv, _normalize_demo_row, _ensemble_demo,
    _normalize_pain_row, _ensemble_pain_binary, _ensemble_pain_aux,
    pain_aux_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_OUT    = PROJECT_ROOT / "model_outputs"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "analysis"

# BRIDGE tasks (from parse_bridge.py)
BRIDGE_TASKS = {
    "MEDIQA":    {"lang": "en", "n_total": 1501, "source": "MEDIQA 2023-chat-A"},
    "MIMIC":     {"lang": "en", "n_total": 5000, "source": "MIMIC-III Outcome.Mortality"},
    "Cantemist": {"lang": "es", "n_total": 1294, "source": "Cantemist-NER"},
    "IFMIR":     {"lang": "ja", "n_total": 5000, "source": "IFMIR-Incident type"},
    "Brateca":   {"lang": "pt", "n_total": 5000, "source": "Brateca-Mortality"},
}


def _load_pain_per_model(task: str) -> Dict[str, List[Dict[str, str]]]:
    out = {}
    for model_name, model_dir in DEMOGRAPHIC_MODEL_DIRS.items():
        path = model_dir / f"BRIDGE_{task}_pain.csv"
        if path.exists():
            out[model_name] = _read_csv(path)
    return out


def _load_demo_per_model(task: str) -> Dict[str, List[Dict[str, str]]]:
    out = {}
    for model_name, model_dir in DEMOGRAPHIC_MODEL_DIRS.items():
        path = model_dir / f"BRIDGE_{task}_demographics.csv"
        if path.exists():
            out[model_name] = _read_csv(path)
    return out


def load_bridge_task(task: str) -> List[Dict[str, Any]]:
    """Row-aligned records for one BRIDGE task: ensemble pain (binary + aux) +
    ensemble demographics. Uses the same 3-model ensemble helpers as
    analysis.py — strict all-same on the binary pain fields, union-with-
    drop-ties on demographics and pain-aux fields (location, pain_type,
    severity, is_mental)."""
    pain_per_model = _load_pain_per_model(task)
    demo_per_model = _load_demo_per_model(task)
    if not pain_per_model or not demo_per_model:
        return []

    n_use = min([len(rs) for rs in pain_per_model.values()] +
                [len(rs) for rs in demo_per_model.values()])
    records = []
    for i in range(n_use):
        # per-model normalised pain + demo rows
        pain_norm = {m: _normalize_pain_row(pain_per_model[m][i])
                     for m in pain_per_model}
        demo_norm = {m: _normalize_demo_row(demo_per_model[m][i])
                     for m in demo_per_model}
        pr_bin   = _ensemble_pain_binary({m: pain_norm[m]["pain_related"]
                                          for m in pain_norm})
        pp_bin   = _ensemble_pain_binary({m: pain_norm[m]["pain_presence"]
                                          for m in pain_norm})
        # Table 1 consensus substrate: pain_presence counts as resolved only
        # when BOTH pain fields are unanimous (drops no pain-positive row).
        if pr_bin is None:
            pp_bin = None
        pain_aux = _ensemble_pain_aux(pain_norm)
        ens_demo = _ensemble_demo(demo_norm)

        records.append({
            "row_idx":           i,
            "language":          BRIDGE_TASKS[task]["lang"],
            "task":              task,
            # Pain ensemble (strict all-same on binary core)
            "pain_related":      pr_bin,
            "pain_presence":     pp_bin,
            # Pain auxiliary ensemble (union-with-drop-ties / per-item vote)
            "location":          pain_aux["location"],
            "pain_type":         pain_aux["pain_type"],
            "severity":          pain_aux["severity"],
            "is_mental":         pain_aux["is_mental"],
            # Demographic ensemble (union-with-drop-ties)
            "age":               ens_demo["age"],
            "age_group":         ens_demo["age_group"],
            "gender":            ens_demo["gender"],
            "ethnicity":         ens_demo["ethnicity"],
            "occupation":        ens_demo["occupation"],
        })
    return records


def _pain_consensus(r: Dict) -> bool:
    """Table 1 consensus substrate: BOTH pain fields ensemble-resolved
    (strict all-same on pain_related AND pain_presence)."""
    return r["pain_related"] is not None and r["pain_presence"] is not None


def _pain_density(records: List[Dict]) -> Dict[str, Any]:
    n_total = len(records)
    n_pp_agreed = sum(1 for r in records if _pain_consensus(r))
    n_pp_true   = sum(1 for r in records if _pain_consensus(r) and r["pain_presence"] is True)
    return {
        "n_total":       n_total,
        "n_pp_agreed":   n_pp_agreed,
        "n_pp_positive": n_pp_true,
        "pp_positive_rate_over_agreed":  n_pp_true / n_pp_agreed if n_pp_agreed else 0.0,
        "pp_positive_rate_over_total":   n_pp_true / n_total      if n_total else 0.0,
    }


def _demo_coverage(records: List[Dict]) -> Dict[str, float]:
    # Computed on the pain-consensus subset (both pain fields resolved), so the
    # whole figure / Table 2 share one substrate with the stratified panels.
    cons = [r for r in records if _pain_consensus(r)]
    n = len(cons)
    fields = ("age", "age_group", "gender", "ethnicity", "occupation")
    return {f: sum(1 for r in cons if r[f] is not None) / n if n else 0.0
            for f in fields}


def _pain_rate_by_field(records: List[Dict], field: str,
                        require_pp_agreed: bool = True) -> Dict[str, Any]:
    """Pain-presence rate stratified by a demographic field.
    Only rows where the field is resolved AND both pain fields are
    ensemble-resolved (Table 1 consensus substrate)."""
    buckets: Dict[str, Dict[str, int]] = {}
    for r in records:
        v = r.get(field)
        if v is None: continue
        if require_pp_agreed and not _pain_consensus(r): continue
        b = buckets.setdefault(str(v), {"pain_present": 0, "total": 0})
        b["total"] += 1
        if r["pain_presence"] is True:
            b["pain_present"] += 1
    # rates
    for b in buckets.values():
        b["rate"] = b["pain_present"] / b["total"] if b["total"] else 0.0
    # chi2 (one row per category, 2 cols: pain_present, pain_absent)
    if len(buckets) >= 2:
        table = [[b["pain_present"], b["total"] - b["pain_present"]]
                 for b in buckets.values()]
        chi2, df, p = chi_square(table)
    else:
        chi2, df, p = 0.0, 0, "n.s."
    return {"groups": buckets, "chi2": chi2, "df": df, "p": p}


def analyse_task(task: str) -> Dict[str, Any]:
    records = load_bridge_task(task)
    if not records:
        return {"error": f"no records for {task}"}
    return {
        "n_total":           len(records),
        "pain_density":      _pain_density(records),
        "coverage":          _demo_coverage(records),
        "pain_aux":          pain_aux_summary(records),
        "gender_x_pain":     _pain_rate_by_field(records, "gender"),
        "age_group_x_pain":  _pain_rate_by_field(records, "age_group"),
        "ethnicity_x_pain":  _pain_rate_by_field(records, "ethnicity"),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, Any] = {"tasks": {}}

    print(f"\n{'BRIDGE task':<12}  {'lang':<4}  {'n':>5}  "
          f"{'pp%(agreed)':>12}  {'cov_age':>7}  {'cov_gen':>7}  "
          f"{'cov_eth':>7}  {'cov_occ':>7}")
    print("-" * 90)
    for task, meta in BRIDGE_TASKS.items():
        r = analyse_task(task)
        if "error" in r:
            print(f"{task:<12}  {meta['lang']:<4}  ERROR: {r['error']}")
            continue
        all_results["tasks"][task] = r
        pd = r["pain_density"]
        cov = r["coverage"]
        print(f"{task:<12}  {meta['lang']:<4}  {r['n_total']:>5}  "
              f"{pd['pp_positive_rate_over_agreed']*100:>11.1f}%  "
              f"{cov['age']*100:>6.1f}%  {cov['gender']*100:>6.1f}%  "
              f"{cov['ethnicity']*100:>6.1f}%  {cov['occupation']*100:>6.1f}%")

    print("\n" + "="*90)
    print("Gender × pain (BRIDGE per task)")
    print("="*90)
    print(f"{'task':<12}  {'female':<20}  {'male':<20}  {'chi2':>7}  {'p':<7}")
    print("-" * 90)
    for task, res in all_results["tasks"].items():
        g = res.get("gender_x_pain", {}).get("groups", {})
        f_str = (f"{g['female']['rate']*100:.1f}% ({g['female']['pain_present']}/{g['female']['total']})"
                 if 'female' in g else "n/a")
        m_str = (f"{g['male']['rate']*100:.1f}% ({g['male']['pain_present']}/{g['male']['total']})"
                 if 'male' in g else "n/a")
        chi2 = res["gender_x_pain"].get("chi2", 0)
        p    = res["gender_x_pain"].get("p", "n.s.")
        print(f"{task:<12}  {f_str:<20}  {m_str:<20}  {chi2:>7.3f}  {p:<7}")

    print("\n" + "="*90)
    print("Age-group × pain (BRIDGE per task)")
    print("="*90)
    print(f"{'task':<12}  {'pediatric':<18}  {'adult':<18}  {'elderly':<18}  {'chi2':>6}  {'p':<7}")
    print("-" * 100)
    for task, res in all_results["tasks"].items():
        g = res.get("age_group_x_pain", {}).get("groups", {})
        def _fmt(k):
            if k not in g: return "n/a"
            return f"{g[k]['rate']*100:.1f}% ({g[k]['pain_present']}/{g[k]['total']})"
        chi2 = res["age_group_x_pain"].get("chi2", 0)
        p    = res["age_group_x_pain"].get("p", "n.s.")
        print(f"{task:<12}  {_fmt('pediatric'):<18}  {_fmt('adult'):<18}  "
              f"{_fmt('elderly'):<18}  {chi2:>6.2f}  {p:<7}")

    print("\n" + "="*100)
    print("Pain auxiliary fields (3-model ensemble; conditional on pain_presence=True)")
    print("="*100)
    print(f"{'task':<12}  {'n_pp':>5}  {'sev_res%':>8}  {'loc_any%':>8}  "
          f"{'type_any%':>9}  {'mental_t%':>9}  top severities")
    print("-"*100)
    for task, res in all_results["tasks"].items():
        pa = res.get("pain_aux", {})
        sev_str = ", ".join(f"{k}:{v}" for k, v in pa.get("severity_dist", {}).items())
        print(f"{task:<12}  {pa.get('n_pain_present',0):>5}  "
              f"{pa.get('severity_resolved_rate',0)*100:>7.1f}%  "
              f"{pa.get('location_any_rate',0)*100:>7.1f}%  "
              f"{pa.get('pain_type_any_rate',0)*100:>8.1f}%  "
              f"{pa.get('is_mental_true_rate',0)*100:>8.1f}%  "
              f"{sev_str}")

    out_path = OUTPUT_DIR / "bridge_pain_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n→ Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
