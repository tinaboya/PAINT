#!/usr/bin/env python3
"""
Calculate evaluation metrics from prediction JSON files
Works for both LLM and Regex results that don't have metrics yet.

Location metrics can be computed in two ways:
- canonical (default): Keyword/canonical mapping. Fast, no extra deps; same body region
  (e.g. "abdominal pain", "right hypochondrium") maps to a canonical form and matches
  are counted by set overlap. Requires maintaining a phrase list.
- semantic: Embedding-based similarity (optional). Install: pip install sentence-transformers.
  Encodes each location string and counts a match when cosine similarity >= threshold.
  Handles paraphrases and synonyms without a keyword list; better for research comparison.
  Use --location-metric semantic when running the CLI.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Lazy-loaded encoder for semantic location matching (only when location_metric="semantic")
_semantic_encoder = None


def _get_semantic_encoder():
    """Load a small sentence encoder once. Returns None if sentence_transformers not installed."""
    global _semantic_encoder
    if _semantic_encoder is not None:
        return _semantic_encoder
    try:
        from sentence_transformers import SentenceTransformer
        _semantic_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return _semantic_encoder
    except Exception:
        return None

def _valid_predictions(predictions: List[Dict]) -> Tuple[List[Dict], int]:
    """Return predictions that have valid structure (no API error). Skip entries where prediction is {'error': ...}."""
    valid = []
    error_count = 0
    for pred in predictions:
        p = pred.get("prediction")
        if not isinstance(p, dict):
            error_count += 1
            continue
        if "error" in p:
            error_count += 1
            continue
        if "pain_related" in p or "pain_presence" in p:
            valid.append(pred)
        else:
            error_count += 1
    return valid, error_count

def calculate_binary_metrics(predictions: List[Dict], field: str) -> Dict:
    """Calculate accuracy, precision, recall, and BOTH binary and macro F1
    for a binary field. Skips samples with prediction errors.

    f1       = positive-class F1 (sklearn default, average='binary')
    f1_macro = (F1_positive + F1_negative) / 2 (sklearn average='macro')
    """
    valid, error_count = _valid_predictions(predictions)
    tp = fp = tn = fn = 0

    for pred in valid:
        gt = pred["ground_truth"][field]
        pred_val = pred["prediction"].get(field, False)

        if gt and pred_val:
            tp += 1
        elif not gt and pred_val:
            fp += 1
        elif not gt and not pred_val:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    # positive-class metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_pos = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    # negative-class metrics
    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
    rec_neg  = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0
    f1_macro = (f1_pos + f1_neg) / 2

    out = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1_pos, 4),
        "f1_macro": round(f1_macro, 4),
    }
    if error_count > 0:
        out["evaluated_n"] = total
        out["skipped_errors"] = error_count
    return out

# Canonical location forms so "abdominal pain" matches "abdominal", and more specific preds (e.g. "lower right abdomen") match GT "abdominal".
# Include multi-word and region terms so LLM (cot) outputs get fair credit.
_LOCATION_CANONICAL = {
    "abdomen": "abdominal",
    "abdominal": "abdominal",
    "hypochondrium": "abdominal",
    "hypochondria": "abdominal",
    "right hypochondrium": "abdominal",
    "left hypochondrium": "abdominal",
    "lower right abdomen": "abdominal",
    "lower left abdomen": "abdominal",
    "right abdomen": "abdominal",
    "left abdomen": "abdominal",
    "lower abdominal": "abdominal",
    "upper abdominal": "abdominal",
    "right upper quadrant": "abdominal",
    "left upper quadrant": "abdominal",
    "ruq": "abdominal",
    "luq": "abdominal",
    "iliac": "abdominal",
    "iliac fossa": "abdominal",
    "chest": "chest",
    "thoracic": "chest",
    "thorax": "chest",
    "hemithorax": "chest",
    "left hemithorax": "chest",
    "right hemithorax": "chest",
    "epigastric": "epigastric",
    "epigastrium": "epigastric",
    "back": "back",
    "lumbar": "back",
    "lower back": "back",
    "low back": "back",
    "head": "head",
    "headache": "head",
    "neck": "neck",
    "cervical": "neck",
    "throat": "throat",
    "leg": "leg",
    "left leg": "leg",
    "right leg": "leg",
    "calf": "leg",
    "thigh": "leg",
    "thighs": "leg",
    "lower limbs": "leg",
    "arm": "arm",
    "left arm": "arm",
    "right arm": "arm",
    "pelvis": "pelvic",
    "pelvic": "pelvic",
    "hip": "hip",
    "knee": "knee",
    "right knee": "knee",
    "left knee": "knee",
    "shoulder": "shoulder",
    "right shoulder": "shoulder",
    "left shoulder": "shoulder",
    "groin": "groin",
    "mental": "mental",
    "emotional": "mental",
}
# Sorted by key length descending so longer phrases match first (e.g. "lower right abdomen" before "abdomen")
_LOCATION_CANONICAL_SORTED = sorted(_LOCATION_CANONICAL.items(), key=lambda x: -len(x[0]))


def _normalize_location_string(s: str) -> str:
    """Lowercase, strip trailing ' pain' / ' pain.', strip leading 'no ', take first meaningful token."""
    if not s or not isinstance(s, str):
        return ""
    s = s.strip().lower()
    if not s:
        return ""
    if s.startswith("no "):
        s = s[3:].strip()
    for suffix in (" pain", " pain.", " pain,", " discomfort", " pressure"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
            break
    return s


def _location_tokens_to_set(locs: List[str]) -> set:
    """Normalize a list of location strings into a set of canonical forms for comparison.
    Uses exact key, then first-word key, then substring match (longest first) so e.g.
    'right hypochondrium' maps to 'abdominal', 'lower right abdomen' to 'abdominal'."""
    out = set()
    for loc in locs:
        if not loc:
            continue
        norm = _normalize_location_string(str(loc))
        if not norm:
            continue
        parts = [p.strip() for p in norm.replace("/", ",").split(",") if p.strip()]
        for part in parts:
            canonical = _LOCATION_CANONICAL.get(part) or _LOCATION_CANONICAL.get(part.split()[0] if part else "")
            if not canonical:
                for key, val in _LOCATION_CANONICAL_SORTED:
                    if key in part:
                        canonical = val
                        break
            out.add(canonical if canonical else part)
    return out


def _location_metrics_semantic(
    predictions: List[Dict],
    threshold: float = 0.65,
) -> Optional[Dict]:
    """
    Location metrics using embedding similarity. Requires sentence_transformers.
    Greedily matches (gt, pred) pairs by cosine similarity >= threshold; counts TP/FP/FN.
    Returns None if encoder not available.
    """
    encoder = _get_semantic_encoder()
    if encoder is None:
        return None
    valid, error_count = _valid_predictions(predictions)
    tp = fp = fn = 0
    import numpy as np
    for pred in valid:
        gt_loc = pred["ground_truth"].get("location", "")
        pred_locs = pred["prediction"].get("location", [])
        if isinstance(gt_loc, str):
            gt_locs = [loc.strip() for loc in gt_loc.split(",") if loc.strip()]
        else:
            gt_locs = [str(gt_loc).strip()] if gt_loc else []
        if not isinstance(pred_locs, list):
            pred_locs = [pred_locs] if pred_locs else []
        pred_locs = [str(loc).strip() for loc in pred_locs if loc]
        if not gt_locs and not pred_locs:
            continue
        if not gt_locs:
            fp += len(pred_locs)
            continue
        if not pred_locs:
            fn += len(gt_locs)
            continue
        all_texts = gt_locs + pred_locs
        embs = encoder.encode(all_texts)
        n_gt = len(gt_locs)
        gt_embs = embs[:n_gt]
        pred_embs = embs[n_gt:]
        sim = np.dot(gt_embs, pred_embs.T) / (
            np.linalg.norm(gt_embs, axis=1, keepdims=True)
            * np.linalg.norm(pred_embs, axis=1, keepdims=True).T
        )
        matched_gt, matched_pred = set(), set()
        for _ in range(min(n_gt, len(pred_locs))):
            best_sim, best_gt, best_p = -1, -1, -1
            for i in range(n_gt):
                if i in matched_gt:
                    continue
                for j in range(len(pred_locs)):
                    if j in matched_pred:
                        continue
                    if sim[i, j] >= threshold and sim[i, j] > best_sim:
                        best_sim, best_gt, best_p = sim[i, j], i, j
            if best_gt < 0:
                break
            matched_gt.add(best_gt)
            matched_pred.add(best_p)
            tp += 1
        fn += n_gt - len(matched_gt)
        fp += len(pred_locs) - len(matched_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    micro = {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
    if error_count > 0:
        micro["skipped_errors"] = error_count
    return {"micro": micro}


def calculate_location_metrics(
    predictions: List[Dict],
    location_metric: str = "canonical",
    semantic_threshold: float = 0.65,
) -> Dict:
    """Calculate location NER metrics. Skips samples with prediction errors.

    location_metric: "canonical" (keyword/canonical set overlap) or "semantic"
        (embedding similarity; requires sentence_transformers). If "semantic"
        and encoder unavailable, falls back to "canonical".
    """
    if location_metric == "semantic":
        out = _location_metrics_semantic(predictions, threshold=semantic_threshold)
        if out is not None:
            return out
    valid, error_count = _valid_predictions(predictions)
    tp = fp = fn = 0
    for pred in valid:
        gt_loc = pred["ground_truth"].get("location", "")
        pred_locs = pred["prediction"].get("location", [])

        if isinstance(gt_loc, str):
            gt_locs = [loc.strip() for loc in gt_loc.split(",") if loc.strip()]
        else:
            gt_locs = [str(gt_loc).strip()] if gt_loc else []

        if not isinstance(pred_locs, list):
            pred_locs = [pred_locs] if pred_locs else []
        pred_locs = [str(loc).strip() for loc in pred_locs if loc]

        gt_set = _location_tokens_to_set(gt_locs)
        pred_set = _location_tokens_to_set(pred_locs)

        tp += len(gt_set & pred_set)
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    micro = {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
    if error_count > 0:
        micro["skipped_errors"] = error_count
    return {"micro": micro}

def calculate_metrics(
    json_path: Path,
    location_metric: str = "canonical",
    semantic_threshold: float = 0.65,
    force: bool = False,
) -> Dict:
    """Calculate all metrics for a result file.
    location_metric: 'canonical' (default) or 'semantic' for location F1.
    force: if True, recompute even when metrics already present."""
    with open(json_path) as f:
        data = json.load(f)
    
    if not force and 'metrics' in data:
        print(f"✓ {json_path.name} already has metrics")
        return data
    
    print(f"⚙️  Calculating metrics for {json_path.name} (location={location_metric})...")
    predictions = data["predictions"]
    _, skipped = _valid_predictions(predictions)
    if skipped > 0:
        print(f"   (Skipping {skipped} predictions with API errors; metrics on {len(predictions) - skipped} samples)")

    metrics = {
        "pain_related": calculate_binary_metrics(predictions, "pain_related"),
        "pain_presence": calculate_binary_metrics(predictions, "pain_presence"),
        "location": calculate_location_metrics(
            predictions,
            location_metric=location_metric,
            semantic_threshold=semantic_threshold,
        ),
    }
    
    data['metrics'] = metrics
    
    # Save updated file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Metrics saved to {json_path.name}")
    return data

def print_summary_table(results: List[Dict]):
    """Print a summary table of all results"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)
    
    header = f"{'Dataset':<15} {'Provider':<10} {'Model':<20} {'Pain F1':<10} {'Presence F1':<12} {'Location F1':<12}"
    print(header)
    print("-"*80)
    
    for result in results:
        dataset = result.get('dataset', 'Unknown')
        provider = result.get('provider', 'Unknown')
        model = result.get('model', 'Unknown')[:18]
        
        metrics = result.get('metrics', {})
        pain_f1 = metrics.get('pain_related', {}).get('f1', 0)
        presence_f1 = metrics.get('pain_presence', {}).get('f1', 0)
        loc_f1 = metrics.get('location', {}).get('micro', {}).get('f1', 0)
        
        row = f"{dataset:<15} {provider:<10} {model:<20} {pain_f1:<10.4f} {presence_f1:<12.4f} {loc_f1:<12.4f}"
        print(row)
    
    print("="*80)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute extraction metrics from prediction JSONs")
    parser.add_argument("files", nargs="+", help="JSON paths or globs (e.g. outputs/runs/llm_*.json)")
    parser.add_argument(
        "--location-metric",
        choices=("canonical", "semantic"),
        default="canonical",
        help="Location F1: canonical (keyword/canonical map) or semantic (embedding similarity; needs sentence-transformers)",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.65,
        help="Min cosine similarity for semantic location match (default 0.65)",
    )
    parser.add_argument("--force", action="store_true", help="Recompute metrics even if file already has them")
    args = parser.parse_args()
    results = []
    for arg in args.files:
        for json_path in Path().glob(arg):
            if json_path.suffix == ".json":
                result = calculate_metrics(
                    json_path,
                    location_metric=args.location_metric,
                    semantic_threshold=args.semantic_threshold,
                    force=args.force,
                )
                results.append(result)
    if results:
        print_summary_table(results)

if __name__ == "__main__":
    main()