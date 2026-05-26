#!/usr/bin/env python3
"""
Majority-vote ensemble + all-same gold-sample extraction for WorldMedQA.

Two field groups are handled:
  - pain:        pain_related, pain_presence (bool)
  - demographic: age (int), gender, occupation, ethnicity (normalized string)

Alignment is by ROW POSITION across all model files (all datasets line up: Brazil
904, Japan 742, Spain 1049). Ground truth for pain comes from WorldMedQA CSVs.
There is no demographic ground truth — demographic results rely on model
agreement only.

Outputs per dataset:
  - ensemble_<dataset>.json   metrics + per-field agreement counts
  - gold_<dataset>_pain.csv   rows where ALL pain models agree on every pain field
  - gold_<dataset>_demographic.csv   same idea for demographic fields
                                     (skipped if <2 demographic models available)

Usage:
    python -m paint.ensemble --all --output outputs/ensemble/
    python -m paint.ensemble --dataset Brazil
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_GT = {
    "Brazil": PROJECT_ROOT / "data" / "raw" / "WorldMedQA" / "Brazil.csv",
    "Japan":  PROJECT_ROOT / "data" / "raw" / "WorldMedQA" / "Japan_T.csv",
    "Spain":  PROJECT_ROOT / "data" / "raw" / "WorldMedQA" / "Spain.csv",
}

# All model prediction CSVs live under model_outputs/<model>/ with the
# uniform naming  {Brazil,Japan,Spain,UK}_{pain,demographics}.csv
MODEL_OUTPUTS = PROJECT_ROOT / "model_outputs"

# Folder name per model. Selected ensemble models first, then the
# closed-source reference (GPT5) and the two un-selected baselines.
MODEL_DIRS = {
    "Gemma4-e4b":      MODEL_OUTPUTS / "gemma4-e4b",
    "Qwen3-8b":        MODEL_OUTPUTS / "qwen3-8b",
    "DeepSeek-R1-32B": MODEL_OUTPUTS / "deepseek-r1-qwen-32b",
    "GPT5":            MODEL_OUTPUTS / "gpt-5",
    "LLaMA-3.1-8B":    MODEL_OUTPUTS / "llama-3.1-8b",
    "MedGemma-1.5":    MODEL_OUTPUTS / "medgemma-1.5",
}


def _build_default_files() -> Dict[str, Dict[str, Dict[str, Path]]]:
    """Build {dataset: {model: {group: path}}}, including only files that
    exist on disk (e.g. GPT5 / LLaMA / MedGemma have no UK partition;
    only DeepSeek has demographic CSVs in this folder layout)."""
    out: Dict[str, Dict[str, Dict[str, Path]]] = {}
    for dataset in DATASET_GT:
        out[dataset] = {}
        for model, mdir in MODEL_DIRS.items():
            entry: Dict[str, Path] = {}
            pain_p = mdir / f"{dataset}_pain.csv"
            demo_p = mdir / f"{dataset}_demographics.csv"
            if pain_p.exists():
                entry["pain"] = pain_p
            if demo_p.exists():
                entry["demographic"] = demo_p
            if entry:
                out[dataset][model] = entry
    return out


# Each entry is {model_name: {group: path}}; missing files are skipped.
DEFAULT_FILES: Dict[str, Dict[str, Dict[str, Path]]] = _build_default_files()

# By default we use the top-3 models the team agreed on for the ensemble.
# GPT5 (closed reference), LLaMA-3.1-8B and MedGemma-1.5 (un-selected
# baselines) are available too but kept out unless explicitly added.
DEFAULT_PAIN_MODELS = ("Gemma4-e4b", "Qwen3-8b", "DeepSeek-R1-32B")

PAIN_FIELDS = ("pain_related", "pain_presence")
DEMOGRAPHIC_FIELDS = ("age", "gender", "occupation", "ethnicity")


# ---------------------------------------------------------------------------
# Field normalization (one helper per field type)
# ---------------------------------------------------------------------------

def _parse_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    v = str(value).strip().upper()
    if v in ("TRUE", "Y", "YES", "1"):
        return True
    if v in ("FALSE", "N", "NO", "0"):
        return False
    return None


_INT_RE = re.compile(r"-?\d+")

def _parse_int(value: Any) -> Optional[int]:
    """Extract the first integer from text (handles '54 anos', '23 years', '54')."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("none", "null", "n/a", "na", "unknown", "nan"):
        return None
    m = _INT_RE.search(s)
    return int(m.group()) if m else None


def _norm_string(value: Any) -> Optional[str]:
    """Lowercase + collapse whitespace + strip punctuation for cross-model string compare."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s or s in ("none", "null", "n/a", "na", "unknown", "nan", "[]", "{}"):
        return None
    s = re.sub(r"[^\w\s]", " ", s)   # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


FIELD_PARSERS = {
    "pain_related":  _parse_bool,
    "pain_presence": _parse_bool,
    "age":           _parse_int,
    "gender":        _norm_string,
    "occupation":    _norm_string,
    "ethnicity":     _norm_string,
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_predictions(path: Path, fields: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Load a model's CSV and extract the requested fields, parsed per field type."""
    rows = _read_csv(path)
    out = []
    for row in rows:
        rec = {}
        for f in fields:
            rec[f] = FIELD_PARSERS[f](row.get(f))
        out.append(rec)
    return out


def load_ground_truth(dataset: str) -> List[Dict[str, Optional[bool]]]:
    """Load WorldMedQA pain GT by row position. No demographic GT available."""
    rows = _read_csv(DATASET_GT[dataset])
    return [{
        "pain_related":  row.get("pain_related (Y/N)", "").strip().upper() == "Y",
        "pain_presence": row.get("pain_presence(Y/N)", "").strip().upper() == "Y",
    } for row in rows]


# ---------------------------------------------------------------------------
# Majority vote (bool) and all-same agreement (generic)
# ---------------------------------------------------------------------------

def majority_vote_bool(votes: List[Optional[bool]]) -> Optional[bool]:
    valid = [v for v in votes if v is not None]
    if not valid:
        return None
    return sum(valid) > len(valid) / 2


def all_same(votes: List[Any]) -> bool:
    """True iff every vote is non-None and they are all equal."""
    if not votes or any(v is None for v in votes):
        return False
    first = votes[0]
    return all(v == first for v in votes)


# ---------------------------------------------------------------------------
# Pain: ensemble + metrics + gold subset
# ---------------------------------------------------------------------------

def _binary_metrics(preds: List[Optional[bool]], labels: List[bool]) -> Dict[str, Any]:
    """Compute confusion matrix + positive-class F1 (binary) + macro F1.

    macro F1 = (F1_positive + F1_negative) / 2, equivalent to
    sklearn.metrics.f1_score(..., average='macro'). It is the
    class-balanced variant used in some clinical-NLP benchmarks.
    """
    tp = fp = tn = fn = skipped = 0
    for p, gt in zip(preds, labels):
        if p is None:
            skipped += 1
            continue
        if p and gt:
            tp += 1
        elif p and not gt:
            fp += 1
        elif not p and not gt:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    # positive class
    precision = tp / (tp + fp)    if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)    if (tp + fn) > 0 else 0.0
    f1_pos    = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    # negative class
    prec_neg  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_neg   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg    = (2 * prec_neg * rec_neg / (prec_neg + rec_neg)
                 if (prec_neg + rec_neg) > 0 else 0.0)
    f1_macro  = (f1_pos + f1_neg) / 2
    return {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1_pos,    4),  # binary (positive-class) F1
        "f1_macro":  round(f1_macro,  4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "skipped": skipped,
    }


def run_pain_ensemble(
    model_preds: Dict[str, List[Dict]],
    ground_truth: List[Dict],
) -> Dict[str, Any]:
    """All-same-agree ensemble for pain fields.

    Gold standard: rows where ALL three models return the same value.
    Reports per-field and all-fields-agree coverage + macro F1 on each subset.
    """
    n = len(ground_truth)
    model_names = list(model_preds)
    for name, preds in model_preds.items():
        if len(preds) != n:
            raise ValueError(f"Pain: model '{name}' has {len(preds)} rows, GT has {n}.")

    out: Dict[str, Any] = {"n": n, "models": {}, "gold": {}}

    # Per-model metrics
    for name in model_names:
        preds = model_preds[name]
        out["models"][name] = {
            f: _binary_metrics([p[f] for p in preds], [g[f] for g in ground_truth])
            for f in PAIN_FIELDS
        }

    # All-same per-field gold + all-fields-agree gold
    per_field_gold_idx: Dict[str, List[int]] = {f: [] for f in PAIN_FIELDS}
    all_fields_gold_idx: List[int] = []
    for i in range(n):
        all_agree_flags = []
        for f in PAIN_FIELDS:
            agreed = all_same([model_preds[m][i][f] for m in model_names])
            all_agree_flags.append(agreed)
            if agreed:
                per_field_gold_idx[f].append(i)
        if all(all_agree_flags):
            all_fields_gold_idx.append(i)

    # Macro F1 on each all-same subset (agreed value = any model's value)
    per_field_agree_metrics: Dict[str, Any] = {}
    for f in PAIN_FIELDS:
        idxs = per_field_gold_idx[f]
        sub_preds  = [model_preds[model_names[0]][i][f] for i in idxs]
        sub_labels = [ground_truth[i][f] for i in idxs]
        per_field_agree_metrics[f] = _binary_metrics(sub_preds, sub_labels)

    all_fields_agree_metrics: Dict[str, Any] = {}
    for f in PAIN_FIELDS:
        sub_preds  = [model_preds[model_names[0]][i][f] for i in all_fields_gold_idx]
        sub_labels = [ground_truth[i][f] for i in all_fields_gold_idx]
        all_fields_agree_metrics[f] = _binary_metrics(sub_preds, sub_labels)

    out["gold"] = {
        "models_used":              model_names,
        "per_field_agree_count":    {f: len(v) for f, v in per_field_gold_idx.items()},
        "per_field_agree_rate":     {f: round(len(v) / n, 4) for f, v in per_field_gold_idx.items()},
        "per_field_agree_metrics":  per_field_agree_metrics,
        "all_fields_agree_count":   len(all_fields_gold_idx),
        "all_fields_agree_rate":    round(len(all_fields_gold_idx) / n, 4),
        "all_fields_agree_metrics": all_fields_agree_metrics,
        "all_fields_gold_idx":      all_fields_gold_idx,
    }
    return out


# ---------------------------------------------------------------------------
# Demographic: agreement-only (no GT) + gold subset
# ---------------------------------------------------------------------------

def run_demographic_agreement(
    model_preds: Dict[str, List[Dict]],
) -> Dict[str, Any]:
    """No ground truth — compute per-field agreement and all-same gold subset."""
    if not model_preds:
        return {"status": "skipped: no demographic predictions available"}

    n = len(next(iter(model_preds.values())))
    model_names = list(model_preds)
    for name, preds in model_preds.items():
        if len(preds) != n:
            raise ValueError(f"Demo: model '{name}' has {len(preds)} rows, expected {n}.")

    # Coverage per model per field (non-null rate)
    coverage = {
        name: {
            f: round(sum(1 for p in preds if p[f] is not None) / n, 4)
            for f in DEMOGRAPHIC_FIELDS
        }
        for name, preds in model_preds.items()
    }

    out: Dict[str, Any] = {
        "n":            n,
        "models_used":  model_names,
        "coverage":     coverage,
    }

    if len(model_names) < 2:
        out["status"] = (
            f"only 1 demographic model ({model_names[0]}) — no cross-model "
            "agreement possible. Pending the 7B model output."
        )
        # We still expose the single model's parsed predictions so the gold
        # CSV can carry demographics alongside pain-gold rows.
        out["single_model_predictions"] = model_preds[model_names[0]]
        return out

    per_field_gold_idx: Dict[str, List[int]] = {f: [] for f in DEMOGRAPHIC_FIELDS}
    all_fields_gold_idx: List[int] = []
    for i in range(n):
        all_agree_flags = []
        for f in DEMOGRAPHIC_FIELDS:
            agreed = all_same([model_preds[m][i][f] for m in model_names])
            all_agree_flags.append(agreed)
            if agreed:
                per_field_gold_idx[f].append(i)
        if all(all_agree_flags):
            all_fields_gold_idx.append(i)

    out["gold"] = {
        "per_field_agree_count":  {f: len(v) for f, v in per_field_gold_idx.items()},
        "per_field_agree_rate":   {f: round(len(v) / n, 4) for f, v in per_field_gold_idx.items()},
        "all_fields_agree_count": len(all_fields_gold_idx),
        "all_fields_agree_rate":  round(len(all_fields_gold_idx) / n, 4),
        "all_fields_gold_idx":    all_fields_gold_idx,
    }
    return out


# ---------------------------------------------------------------------------
# Gold-sample CSV writers
# ---------------------------------------------------------------------------

def write_pain_gold_csv(
    out_path: Path,
    pain_result: Dict[str, Any],
    model_preds: Dict[str, List[Dict]],
    demographic_preds: Optional[Dict[str, List[Dict]]],
    gt_rows: List[Dict],
) -> int:
    """Write one row per index where every model agreed on every pain field.

    Columns: row_idx, agreed_pain_related, agreed_pain_presence,
             gt_pain_related, gt_pain_presence,
             (optional) <model>_<demo_field> for the single available demo model.
    """
    idxs = pain_result["gold"]["all_fields_gold_idx"]
    model_names = pain_result["gold"]["models_used"]

    demo_model = next(iter(demographic_preds)) if demographic_preds else None

    fieldnames = ["row_idx"]
    fieldnames += [f"agreed_{f}" for f in PAIN_FIELDS]
    fieldnames += [f"gt_{f}" for f in PAIN_FIELDS]
    if demo_model:
        fieldnames += [f"{demo_model}_{f}" for f in DEMOGRAPHIC_FIELDS]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in idxs:
            # All models agree → take the first model's value as the agreed value.
            agreed = {fld: model_preds[model_names[0]][i][fld] for fld in PAIN_FIELDS}
            row = {"row_idx": i}
            for fld in PAIN_FIELDS:
                row[f"agreed_{fld}"] = agreed[fld]
                row[f"gt_{fld}"] = gt_rows[i][fld]
            if demo_model:
                for fld in DEMOGRAPHIC_FIELDS:
                    row[f"{demo_model}_{fld}"] = demographic_preds[demo_model][i][fld]
            writer.writerow(row)
    return len(idxs)


def write_demographic_gold_csv(
    out_path: Path,
    demo_result: Dict[str, Any],
    model_preds: Dict[str, List[Dict]],
) -> int:
    """Write one row per index where every demo model agreed on every demo field.
    Skipped if fewer than 2 demographic models were available."""
    if "gold" not in demo_result:
        return 0
    idxs = demo_result["gold"]["all_fields_gold_idx"]
    model_names = demo_result["models_used"]

    fieldnames = ["row_idx"] + [f"agreed_{f}" for f in DEMOGRAPHIC_FIELDS]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in idxs:
            row = {"row_idx": i}
            for fld in DEMOGRAPHIC_FIELDS:
                row[f"agreed_{fld}"] = model_preds[model_names[0]][i][fld]
            writer.writerow(row)
    return len(idxs)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _fmt(m: Dict) -> str:
    return (f"Acc={m['accuracy']:.3f}  P={m['precision']:.3f}  "
            f"R={m['recall']:.3f}  F1={m['f1']:.3f}")


def print_results(dataset: str, pain_result: Dict, demo_result: Dict) -> None:
    n = pain_result["n"]
    pain_models = pain_result["gold"]["models_used"]
    print(f"\n{'='*68}")
    print(f"  Dataset: {dataset}  |  n={n}")
    print(f"  Pain models:        {', '.join(pain_models)}")
    print(f"  Demographic models: {', '.join(demo_result.get('models_used', [])) or '(none)'}")
    print(f"{'='*68}")

    print("\n  [Pain] per-model vs WorldMedQA GT:")
    for fld in PAIN_FIELDS:
        print(f"\n    {fld}:")
        for name in pain_models:
            print(f"      {name:<18} {_fmt(pain_result['models'][name][fld])}")

    g = pain_result["gold"]
    print("\n  [Pain] all-same gold subset (all 3 models agree):")
    for fld in PAIN_FIELDS:
        cnt  = g["per_field_agree_count"][fld]
        rate = g["per_field_agree_rate"][fld]
        m    = g["per_field_agree_metrics"][fld]
        print(f"      {fld:<18} {cnt}/{n}  ({rate:.1%})  {_fmt(m)}")
    cnt_all  = g["all_fields_agree_count"]
    rate_all = g["all_fields_agree_rate"]
    print(f"\n      ALL pain fields   {cnt_all}/{n}  ({rate_all:.1%})")
    for fld in PAIN_FIELDS:
        m = g["all_fields_agree_metrics"][fld]
        print(f"        {fld:<16} {_fmt(m)}")

    print("\n  [Demographic]:")
    if "gold" not in demo_result:
        print(f"      {demo_result.get('status', '(no demo result)')}")
        if "coverage" in demo_result:
            for name, cov in demo_result["coverage"].items():
                cov_str = "  ".join(f"{f}={r:.1%}" for f, r in cov.items())
                print(f"      coverage {name:<18} {cov_str}")
    else:
        g = demo_result["gold"]
        for fld in DEMOGRAPHIC_FIELDS:
            cnt  = g["per_field_agree_count"][fld]
            rate = g["per_field_agree_rate"][fld]
            print(f"      {fld:<18} {cnt}/{n}  ({rate:.1%})")
        print(f"      {'ALL demo fields':<18} {g['all_fields_agree_count']}/{n}  ({g['all_fields_agree_rate']:.1%})")


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------

def run_dataset(
    dataset: str,
    pain_model_files: Dict[str, Path],
    demo_model_files: Dict[str, Path],
    out_dir: Optional[Path],
) -> Tuple[Dict, Dict]:
    available_pain = {n: p for n, p in pain_model_files.items() if p.exists()}
    available_demo = {n: p for n, p in demo_model_files.items() if p.exists()}

    missing_pain = set(pain_model_files) - set(available_pain)
    missing_demo = set(demo_model_files) - set(available_demo)
    if missing_pain:
        print(f"[{dataset}] Missing pain files: {', '.join(missing_pain)}")
    if missing_demo:
        print(f"[{dataset}] Missing demo files: {', '.join(missing_demo)}")
    if not available_pain:
        print(f"[{dataset}] No pain prediction files found, skipping dataset.")
        return {}, {}

    pain_preds = {n: load_predictions(p, PAIN_FIELDS)         for n, p in available_pain.items()}
    demo_preds = {n: load_predictions(p, DEMOGRAPHIC_FIELDS)  for n, p in available_demo.items()}
    gt_rows = load_ground_truth(dataset)

    pain_result = run_pain_ensemble(pain_preds, gt_rows)
    demo_result = run_demographic_agreement(demo_preds)

    print_results(dataset, pain_result, demo_result)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON summary (strip large index lists / raw predictions)
        pain_json = {**pain_result, "gold": {
            k: v for k, v in pain_result["gold"].items()
            if k != "all_fields_gold_idx"
        }}
        demo_json = {k: v for k, v in demo_result.items()
                     if k != "single_model_predictions"}
        if "gold" in demo_json:
            demo_json["gold"] = {k: v for k, v in demo_json["gold"].items()
                                 if k != "all_fields_gold_idx"}

        summary = {"dataset": dataset, "pain": pain_json, "demographic": demo_json}
        json_path = out_dir / f"ensemble_{dataset}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  → metrics JSON: {json_path}")

        # Gold CSVs
        pain_gold_path = out_dir / f"gold_{dataset}_pain.csv"
        n_pain = write_pain_gold_csv(
            pain_gold_path, pain_result, pain_preds, demo_preds, gt_rows,
        )
        print(f"  → pain gold CSV ({n_pain} rows): {pain_gold_path}")

        if "gold" in demo_result:
            demo_gold_path = out_dir / f"gold_{dataset}_demographic.csv"
            n_demo = write_demographic_gold_csv(demo_gold_path, demo_result, demo_preds)
            print(f"  → demo gold CSV ({n_demo} rows): {demo_gold_path}")

    return pain_result, demo_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_files(
    dataset: str,
    pain_models: Tuple[str, ...],
) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """Pick pain + demographic file paths for the requested pain models."""
    cfg = DEFAULT_FILES[dataset]
    pain_files = {m: cfg[m]["pain"] for m in pain_models if m in cfg and "pain" in cfg[m]}
    demo_files = {m: cfg[m]["demographic"]
                  for m, group in cfg.items() if "demographic" in group}
    return pain_files, demo_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Majority-vote ensemble + all-same gold sample extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", "-d", choices=list(DATASET_GT),
                        help="Single dataset to evaluate (Brazil, Japan, or Spain)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Run on all three datasets")
    parser.add_argument("--pain-models", nargs="+", default=list(DEFAULT_PAIN_MODELS),
                        help=(f"Subset of pain models to ensemble. "
                              f"Default: {list(DEFAULT_PAIN_MODELS)}. "
                              f"GPT5 is also available."))
    parser.add_argument("--output", "-o",
                        help="Directory to save JSON + gold CSVs (optional)")
    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.error("Specify --dataset <name> or --all")

    datasets = list(DATASET_GT) if args.all else [args.dataset]
    out_dir  = Path(args.output) if args.output else None
    pain_models = tuple(args.pain_models)

    for ds in datasets:
        pain_files, demo_files = _resolve_files(ds, pain_models)
        run_dataset(ds, pain_files, demo_files, out_dir)

    print()


if __name__ == "__main__":
    main()
