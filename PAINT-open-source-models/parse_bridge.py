"""Parse BRIDGE_result raw LLM-output JSONs into WMQA-format CSVs.

Uses the same parse_utils.parse_pain / parse_demographics as the WorldMedQA
inference pipeline, to keep extraction behavior consistent across the
two splits (use one shared parser for both).

Input layout:
    BRIDGE_result/
        {pain|demo}.{lang}.{task}/
            {model-name}/
                {task-tag}.result.json   (list of records with `pred` field)

Output layout (matches existing model_outputs/<model>/<Dataset>_pain.csv):
    model_outputs/<model>/BRIDGE_<TaskShort>_<pain|demographics>.csv

CSV schema (matches existing WMQA-Pain CSVs):
    pain:  Question_ID, ok, pain_related, pain_presence, location,
           pain_type, severity, is_mental, context
    demo:  row_index, ok, age, gender, ethnicity, occupation
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Import the official WMQA parser
from parse_utils import parse_pain, parse_demographics

# -----------------------------------------------------------------------------
# Paths + mappings
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
BRIDGE_IN = REPO_ROOT / "BRIDGE_result"
MODEL_OUT = REPO_ROOT / "model_outputs"

# BRIDGE_result subfolder model name → model_outputs/ subfolder name
MODEL_NAME_MAP = {
    "gemma-4-E4B-it":                "gemma4-e4b",
    "Qwen3-8B-Thinking":             "qwen3-8b",
    "DeepSeek-R1-Distill-Qwen-32B":  "deepseek-r1-qwen-32b",
}

# BRIDGE task-folder name → short tag used in output CSV filename
TASK_SHORT_MAP = {
    "MEDIQA 2023-chat-A":         "MEDIQA",
    "MIMIC-III Outcome.Mortality":"MIMIC",
    "Cantemis-NER":               "Cantemist",
    "IFMIR-Incident type":        "IFMIR",
    "Brateca-Mortality":          "Brateca",
}

PAIN_FIELDS = ["pain_related", "pain_presence", "location", "pain_type",
               "severity", "is_mental", "context"]
DEMO_FIELDS = ["age", "gender", "ethnicity", "occupation"]


def _to_bool_str(v):
    if v is True:  return "TRUE"
    if v is False: return "FALSE"
    return ""


def _to_list_str(v):
    """Serialize lists as semicolon-separated (matches WMQA CSV convention)."""
    if isinstance(v, list):
        return ";".join(str(x) for x in v)
    return str(v) if v else ""


def _to_str(v):
    if v is None: return ""
    return str(v)


def _to_pain_row(idx: int, parsed: dict | None, error: str) -> dict:
    """parsed = output of parse_utils.parse_pain (validated dict or None);
    error = '' on full success or description string otherwise."""
    ok = "TRUE" if parsed is not None and not error else "FALSE"
    if parsed is None:
        return {"Question_ID": idx, "ok": ok,
                **{f: "" for f in PAIN_FIELDS}}
    return {
        "Question_ID":   idx,
        "ok":            ok,
        "pain_related":  _to_bool_str(parsed.get("pain_related")),
        "pain_presence": _to_bool_str(parsed.get("pain_presence")),
        "location":      _to_list_str(parsed.get("location")),
        "pain_type":     _to_list_str(parsed.get("pain_type")),
        "severity":      _to_str(parsed.get("severity")),
        "is_mental":     _to_bool_str(parsed.get("is_mental")),
        "context":       _to_str(parsed.get("context")),
    }


def _to_demo_row(idx: int, parsed: dict | None, error: str) -> dict:
    ok = "TRUE" if parsed is not None and not error else "FALSE"
    if parsed is None:
        return {"row_index": idx, "ok": ok,
                **{f: "" for f in DEMO_FIELDS}}
    return {
        "row_index": idx,
        "ok":        ok,
        "age":       _to_str(parsed.get("age")),
        "gender":    _to_str(parsed.get("gender")),
        "ethnicity": _to_str(parsed.get("ethnicity")),
        "occupation":_to_str(parsed.get("occupation")),
    }


def process_one_file(json_path: Path, kind: str,
                     out_csv: Path) -> dict:
    """Parse one BRIDGE_result JSON file and write a CSV.
    Returns a stats dict.
    """
    parser_fn = parse_pain if kind == "pain" else parse_demographics
    row_fn    = _to_pain_row if kind == "pain" else _to_demo_row
    fieldnames = (["Question_ID", "ok"] + PAIN_FIELDS) if kind == "pain" \
                 else (["row_index", "ok"] + DEMO_FIELDS)

    with open(json_path) as f:
        data = json.load(f)

    n_parse_ok  = 0     # extracted FINAL_JSON cleanly
    n_valid_ok  = 0     # parse_ok AND validate_ok
    n_total     = len(data)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for i, rec in enumerate(data):
            pred_text = rec.get("pred", "") or ""
            parsed, error = parser_fn(pred_text) if pred_text \
                            else (None, "empty pred")
            if parsed is not None:
                n_parse_ok += 1
                if not error:
                    n_valid_ok += 1
            w.writerow(row_fn(i, parsed, error))

    return {
        "n_total":      n_total,
        "n_parse_ok":   n_parse_ok,
        "n_valid_ok":   n_valid_ok,
        "parse_rate":   n_parse_ok / n_total if n_total else 0.0,
        "valid_rate":   n_valid_ok / n_total if n_total else 0.0,
    }


def iter_bridge_files():
    """Yield (kind, lang, task_short, model_in_name, model_out_name, json_path)."""
    for task_dir in sorted(BRIDGE_IN.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        # Folder name pattern: "{pain|demo}.{lang}.{task}"
        parts = task_dir.name.split(".", 2)
        if len(parts) != 3:
            continue
        kind_pfx, lang, task_full = parts
        kind = "pain" if kind_pfx == "pain" else "demographics"
        task_short = TASK_SHORT_MAP.get(task_full, task_full.replace(" ", "_"))
        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            mapped = MODEL_NAME_MAP.get(model_dir.name)
            if not mapped:
                continue
            for jf in model_dir.glob("*.json"):
                yield kind, lang, task_short, model_dir.name, mapped, jf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="filter by task short tag (e.g. Cantemist)")
    ap.add_argument("--model", help="filter by output model name (e.g. gemma4-e4b)")
    ap.add_argument("--kind",  choices=("pain","demographics"),
                    help="filter by kind")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't write CSVs, just report parse rates")
    args = ap.parse_args()

    print(f"{'kind':<13} {'lang':<5} {'task':<10} {'model':<22} "
          f"{'n':>5} {'parse%':>7} {'valid%':>7}  output")
    print("-" * 110)
    for kind, lang, task_short, model_in, model_out, jf in iter_bridge_files():
        if args.task  and task_short != args.task:  continue
        if args.model and model_out != args.model:  continue
        if args.kind  and kind != args.kind:        continue
        out_csv = MODEL_OUT / model_out / f"BRIDGE_{task_short}_{kind}.csv"
        if args.dry_run:
            # just count
            with open(jf) as f: n = len(json.load(f))
            stats = {"n_total": n, "parse_rate": 0, "valid_rate": 0}
            mark = "[DRY]"
        else:
            stats = process_one_file(jf, kind, out_csv)
            mark = "→"
        print(f"{kind:<13} {lang:<5} {task_short:<10} {model_out:<22} "
              f"{stats['n_total']:>5} "
              f"{stats['parse_rate']*100:>6.1f}% "
              f"{stats['valid_rate']*100:>6.1f}% "
              f" {mark} {out_csv.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
