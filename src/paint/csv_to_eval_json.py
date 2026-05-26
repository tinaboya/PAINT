#!/usr/bin/env python3
"""
Convert CSV-format model predictions into the JSON schema consumed by
`paint.extraction_evaluation` so that all open-weight models (Gemma4,
Qwen3, DeepSeek) are evaluated with the same canonical evaluator used for
the GPT5 / Gemini runs in `outputs/runs/`.

Alignment: row position (all WorldMedQA + open-weight CSVs already line up:
Brazil 904 / Japan 742 / Spain 1049).

Usage:
    python -m paint.csv_to_eval_json --output outputs/runs/
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_OUTPUTS = PROJECT_ROOT / "model_outputs"
WMQA_DIR      = PROJECT_ROOT / "data" / "raw" / "WorldMedQA"

DATASETS = {
    "Brazil": {"gt": WMQA_DIR / "Brazil.csv",  "lang": "pt"},
    "Japan":  {"gt": WMQA_DIR / "Japan_T.csv", "lang": "ja"},
    "Spain":  {"gt": WMQA_DIR / "Spain.csv",   "lang": "es"},
    "UK":     {"gt": WMQA_DIR / "UK_T.csv",    "lang": "en"},
}

# Folder name per model under model_outputs/. Pain CSVs are named
# {Dataset}_pain.csv. GPT5 / LLaMA / MedGemma have no UK partition;
# missing files are skipped at conversion time.
MODEL_DIRS = {
    "Gemma4-e4b":            MODEL_OUTPUTS / "gemma4-e4b",
    "Qwen3-8b":              MODEL_OUTPUTS / "qwen3-8b",
    "DeepSeek-R1-Qwen-32b":  MODEL_OUTPUTS / "deepseek-r1-qwen-32b",
    "GPT5":                  MODEL_OUTPUTS / "gpt-5",
    "LLaMA-3.1-8B":          MODEL_OUTPUTS / "llama-3.1-8b",
    "MedGemma-1.5":          MODEL_OUTPUTS / "medgemma-1.5",
}

MODEL_FILES = {
    model: {ds: mdir / f"{ds}_pain.csv" for ds in DATASETS}
    for model, mdir in MODEL_DIRS.items()
}


def _parse_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    v = str(value).strip().upper()
    if v in ("TRUE", "Y", "YES", "1"):  return True
    if v in ("FALSE", "N", "NO", "0"):  return False
    return None


def _parse_list(value: Any) -> List[str]:
    if not value:
        return []
    s = str(value).strip()
    if not s:
        return []
    # split on common separators
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    return parts


def load_gt_rows(path: Path) -> List[Dict]:
    """Load WorldMedQA ground-truth rows."""
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_pred_rows(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


_GT_TRUE  = {"Y", "YES"}
_GT_FALSE = {"N", "NO"}


def _gt_label(value: Any) -> Optional[bool]:
    """Accept either ``Y/N`` (Brazil/Japan/Spain) or ``Yes/No`` (UK) GT labels.
    Returns ``None`` for unlabelled rows (empty cell).
    """
    v = (value or "").strip().upper()
    if v in _GT_TRUE:  return True
    if v in _GT_FALSE: return False
    return None


def build_predictions(
    gt_rows: List[Dict],
    pred_rows: List[Dict],
) -> List[Dict]:
    n = min(len(gt_rows), len(pred_rows))
    out = []
    for i in range(n):
        gt   = gt_rows[i]
        pred = pred_rows[i]
        pr_gt = _gt_label(gt.get("pain_related (Y/N)"))
        pp_gt = _gt_label(gt.get("pain_presence(Y/N)"))
        # Skip unlabelled rows (relevant for UK where only ~45/268 carry GT).
        if pr_gt is None or pp_gt is None:
            continue
        out.append({
            "id": str(i),
            "ground_truth": {
                "pain_related":  pr_gt,
                "pain_presence": pp_gt,
                "location":      (gt.get("pain_location") or "").strip(),
            },
            "prediction": {
                "pain_related":  _parse_bool(pred.get("pain_related")),
                "pain_presence": _parse_bool(pred.get("pain_presence")),
                "location":      _parse_list(pred.get("location")),
                "pain_type":     _parse_list(pred.get("pain_type")),
                "severity":      (pred.get("severity") or "").strip() or None,
                "is_mental":     _parse_bool(pred.get("is_mental")),
                "context":       (pred.get("context") or "").strip() or None,
            },
        })
    return out


def convert_one(model: str, dataset: str, out_dir: Path) -> Path:
    csv_path = MODEL_FILES[model][dataset]
    gt_path  = DATASETS[dataset]["gt"]
    lang     = DATASETS[dataset]["lang"]
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    gt_rows   = load_gt_rows(gt_path)
    pred_rows = load_pred_rows(csv_path)
    preds     = build_predictions(gt_rows, pred_rows)

    payload = {
        "dataset":       dataset,
        "provider":      "local",
        "model":         model,
        "prompt_type":   f"cot-native-{lang}",
        "language_mode": "original",
        "total":         len(preds),
        "predictions":   preds,
    }

    out_path = out_dir / f"llm_{model.lower()}_cot-native-{lang}_{dataset}_original.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="outputs/runs",
                        help="Directory to write the JSON files into")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in MODEL_FILES:
        for dataset in DATASETS:
            try:
                p = convert_one(model, dataset, out_dir)
                print(f"  wrote {p}")
            except FileNotFoundError as e:
                print(f"  SKIP {model}/{dataset}: file missing ({e})")


if __name__ == "__main__":
    main()
