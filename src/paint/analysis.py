#!/usr/bin/env python3
"""
PAINT bias analysis — WorldMedQA-Pain (Brazil / Japan / Spain / UK).

All numbers use the 3-model open-weight ensemble
(Gemma4-E4B + Qwen3-8B-Thinking + DeepSeek-R1-Distill-Qwen-32B):
  - pain_related / pain_presence : strict all-same agreement (high precision)
  - is_mental, severity          : majority-with-drop-ties
  - location, pain_type          : per-item majority vote (≥2 of 3 models)
  - demographics                 : union-with-drop-ties

Dimensions:
  1. Pain prevalence: ensemble vs physician GT (confusion + F1)
  2. Per-model + ensemble demographic coverage
  3. Demographic distribution (all rows vs pain-positive subset)
  4. Pain rate stratified by demographic (gender / age_group / ethnicity)
  5. Cross-country comparison + chi-square tests for bias signal

Usage:
    python -m paint.analysis --output outputs/analysis/
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# All model prediction CSVs live under model_outputs/<model>/ with the
# uniform naming  {Brazil,Japan,Spain,UK}_{pain,demographics}.csv
MODEL_OUTPUTS = PROJECT_ROOT / "model_outputs"
WMQA_DIR      = PROJECT_ROOT / "data" / "raw" / "WorldMedQA"

# The 3-model ensemble. Each folder contains both
# {Brazil,Japan,Spain,UK}_pain.csv and {…}_demographics.csv, plus
# BRIDGE_<task>_{pain,demographics}.csv for the BRIDGE-Pain split.
DEMOGRAPHIC_MODEL_DIRS = {
    "DeepSeek-R1-Qwen-32B": MODEL_OUTPUTS / "deepseek-r1-qwen-32b",
    "Gemma4-e4b":           MODEL_OUTPUTS / "gemma4-e4b",
    "Qwen3-8B":             MODEL_OUTPUTS / "qwen3-8b",
}

WMQA_FILES = {
    "Brazil": {"lang": "pt", "gt": WMQA_DIR / "Brazil.csv"},
    "Japan":  {"lang": "ja", "gt": WMQA_DIR / "Japan_T.csv"},
    "Spain":  {"lang": "es", "gt": WMQA_DIR / "Spain.csv"},
    "UK":     {"lang": "en", "gt": WMQA_DIR / "UK_T.csv"},
}


# ---------------------------------------------------------------------------
# Cross-lingual normalization
# ---------------------------------------------------------------------------

GENDER_MAP = {
    # English
    "male": "male", "female": "female", "m": "male", "f": "female",
    "boy": "male", "girl": "female", "man": "male", "woman": "female",
    # Portuguese
    "homem": "male", "menino": "male", "masculino": "male", "rapaz": "male",
    "mulher": "female", "menina": "female", "feminino": "female", "rapariga": "female",
    "criança": "unknown",  # ambiguous unless paired with sex
    # Spanish
    "varon": "male", "varón": "male", "hombre": "male", "niño": "male", "chico": "male",
    "mujer": "female", "niña": "female", "chica": "female", "señora": "female", "señor": "male",
    # Japanese
    "男": "male", "男性": "male", "男子": "male", "男児": "male", "少年": "male",
    "女": "female", "女性": "female", "女子": "female", "女児": "female", "少女": "female",
}

ETHNICITY_MAP = {
    # PT (Brazilian skin-color categories)
    "branca": "white", "branco": "white", "white": "white", "blanca": "white", "blanco": "white",
    "parda": "mixed", "pardo": "mixed", "mestiza": "mixed", "mestizo": "mixed", "mulato": "mixed", "mulata": "mixed",
    "preta": "black", "preto": "black", "negra": "black", "negro": "black", "black": "black", "afro": "black",
    "amarela": "asian", "amarelo": "asian", "asiática": "asian", "asiatica": "asian", "asian": "asian",
    "indígena": "indigenous", "indigena": "indigenous", "índia": "indigenous", "india": "indigenous",
    # JA
    "日本人": "asian", "アジア人": "asian",
    # ES generic
    "hispana": "hispanic", "hispano": "hispanic", "hispanic": "hispanic", "latina": "hispanic", "latino": "hispanic",
    "caucásica": "white", "caucásico": "white", "caucasian": "white",
}


_INT_RE = re.compile(r"-?\d+")

def parse_age(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s or s in {"none", "null", "n/a", "na", "nan", "unknown", ""}:
        return None
    # detect month/day units → child
    if any(u in s for u in ("mes", "month", "día", "dia", "day", "ヶ月", "か月", "日")):
        m = _INT_RE.search(s)
        if m:
            # under a year → store 0
            return 0
    m = _INT_RE.search(s)
    return int(m.group()) if m else None


def age_group(age: Optional[int]) -> Optional[str]:
    if age is None:
        return None
    if age < 18:  return "pediatric"
    if age < 65:  return "adult"
    return "elderly"


def normalize_gender(raw: Any) -> Optional[str]:
    if not raw:
        return None
    s = str(raw).strip().lower()
    if not s or s in {"none", "null", "n/a", "na", "nan", "unknown"}:
        return None
    if s in GENDER_MAP:
        return GENDER_MAP[s]
    # token-by-token
    for tok in re.split(r"[\s,/\-]+", s):
        if tok in GENDER_MAP:
            return GENDER_MAP[tok]
    # JA single-character match
    for ch in s:
        if ch in GENDER_MAP:
            return GENDER_MAP[ch]
    return None


def normalize_ethnicity(raw: Any) -> Optional[str]:
    if not raw:
        return None
    s = str(raw).strip().lower()
    if not s or s in {"none", "null", "n/a", "na", "nan", "unknown"}:
        return None
    if s in ETHNICITY_MAP:
        return ETHNICITY_MAP[s]
    for tok in re.split(r"[\s,/\-]+", s):
        if tok in ETHNICITY_MAP:
            return ETHNICITY_MAP[tok]
    return None


SEVERITY_MAP = {
    # English
    "mild": "mild", "moderate": "moderate", "severe": "severe",
    "low": "mild", "high": "severe", "intense": "severe",
    # Portuguese
    "leve": "mild", "moderada": "moderate", "moderado": "moderate",
    "grave": "severe", "severa": "severe", "severo": "severe", "intensa": "severe", "intenso": "severe",
    # Spanish
    "ligera": "mild", "ligero": "mild", "suave": "mild",
    # Japanese
    "軽度": "mild", "軽い": "mild", "中等度": "moderate", "中程度": "moderate",
    "重度": "severe", "重い": "severe", "激しい": "severe",
}


def normalize_severity(raw: Any) -> Optional[str]:
    """Map free-text severity to {mild, moderate, severe}; None otherwise."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s or s in {"none", "null", "n/a", "na", "nan", "unknown", ""}:
        return None
    if s in SEVERITY_MAP:
        return SEVERITY_MAP[s]
    for tok in re.split(r"[\s,/\-]+", s):
        if tok in SEVERITY_MAP:
            return SEVERITY_MAP[tok]
    return None


_LIST_SEP_RE = re.compile(r"[;,]\s*")

def _parse_list_field(raw: Any) -> Optional[List[str]]:
    """Split a CSV list cell on `;` or `,`, lowercase+strip each item.
    Returns None on empty/missing, otherwise a list (possibly empty if all items
    normalised away)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in {"none", "null", "n/a", "na", "nan", "[]"}:
        return None
    items = []
    for tok in _LIST_SEP_RE.split(s):
        t = tok.strip().strip("[]'\"").strip().lower()
        t = re.sub(r"\s+", " ", t)
        if t and t not in {"none", "null", "n/a"}:
            items.append(t)
    return items if items else None


def parse_bool(raw: Any) -> Optional[bool]:
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        return raw
    v = str(raw).strip().upper()
    if v in ("TRUE", "Y", "YES", "1"): return True
    if v in ("FALSE", "N", "NO", "0"): return False
    return None


# ---------------------------------------------------------------------------
# Chi-square 2xK contingency (df = K-1) — hand-rolled to avoid scipy
# ---------------------------------------------------------------------------

# Critical values for chi-square distribution (df 1..5) at alpha = 0.05/0.01/0.001
_CHI2_CRIT = {
    1: {0.05: 3.841, 0.01: 6.635, 0.001: 10.828},
    2: {0.05: 5.991, 0.01: 9.210, 0.001: 13.816},
    3: {0.05: 7.815, 0.01: 11.345, 0.001: 16.266},
    4: {0.05: 9.488, 0.01: 13.277, 0.001: 18.467},
    5: {0.05: 11.070, 0.01: 15.086, 0.001: 20.515},
}

def chi_square(table: List[List[int]]) -> Tuple[float, int, str]:
    """Pearson chi-square test on a contingency table.

    Returns (chi2, df, p_label).  p_label is one of "<0.001", "<0.01", "<0.05", "n.s.".
    """
    rows = len(table)
    cols = len(table[0]) if rows else 0
    if rows < 2 or cols < 2:
        return 0.0, 0, "n.s."

    row_totals = [sum(r) for r in table]
    col_totals = [sum(table[i][j] for i in range(rows)) for j in range(cols)]
    n = sum(row_totals)
    if n == 0:
        return 0.0, 0, "n.s."

    chi2 = 0.0
    for i in range(rows):
        for j in range(cols):
            exp = row_totals[i] * col_totals[j] / n
            if exp > 0:
                chi2 += (table[i][j] - exp) ** 2 / exp

    df = (rows - 1) * (cols - 1)
    crit = _CHI2_CRIT.get(df, _CHI2_CRIT[5])
    if chi2 >= crit[0.001]: p = "<0.001"
    elif chi2 >= crit[0.01]: p = "<0.01"
    elif chi2 >= crit[0.05]: p = "<0.05"
    else: p = "n.s."
    return round(chi2, 3), df, p


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _gt_bool(raw: Any) -> Optional[bool]:
    """Accept both Y/N (Brazil/Japan/Spain) and Yes/No (UK) GT labels."""
    v = (raw or "").strip().upper()
    if v in ("Y", "YES"):  return True
    if v in ("N", "NO"):   return False
    return None  # unlabelled


def _normalize_demo_row(d: Dict[str, str]) -> Dict[str, Any]:
    """Apply cross-lingual normalisation to one demographic row.
    Occupation: lowercase + strip + collapse whitespace + trim trailing punctuation.
    This dedups surface variants like 'Retired' vs 'retired' and
    'branca' vs 'branca.' that would otherwise break strict all-same agreement.
    """
    age = parse_age(d.get("age"))
    occ_raw = (d.get("occupation") or "").strip().lower()
    occ_raw = re.sub(r"\s+", " ", occ_raw).strip(" .,;:")
    occ_raw = occ_raw or None
    return {
        "raw_age":         d.get("age"),
        "raw_gender":      d.get("gender"),
        "raw_ethnicity":   d.get("ethnicity"),
        "raw_occupation":  d.get("occupation"),
        "age":             age,
        "age_group":       age_group(age),
        "gender":          normalize_gender(d.get("gender")),
        "ethnicity":       normalize_ethnicity(d.get("ethnicity")),
        "occupation":      occ_raw,
    }


_ENSEMBLE_FIELDS = ("age", "age_group", "gender", "ethnicity", "occupation")


def _normalize_pain_row(d: Dict[str, str]) -> Dict[str, Any]:
    """Normalise one pain-CSV row → canonical fields for ensembling.
      - pain_related, pain_presence, is_mental : bool / None
      - location, pain_type                    : list[str] or None
      - severity                               : {mild,moderate,severe} or None
      - context                                : free text (kept raw)
    """
    return {
        "pain_related":  parse_bool(d.get("pain_related")),
        "pain_presence": parse_bool(d.get("pain_presence")),
        "location":      _parse_list_field(d.get("location")),
        "pain_type":     _parse_list_field(d.get("pain_type")),
        "severity":      normalize_severity(d.get("severity")),
        "is_mental":     parse_bool(d.get("is_mental")),
        "context":       (d.get("context") or "").strip() or None,
    }


def _ensemble_pain_binary(per_model_vals: Dict[str, Optional[bool]]) -> Optional[bool]:
    """Strict all-same agreement for the core pain binary fields
    (pain_related, pain_presence). Returns None if any model is None or
    if values disagree. This is the high-precision rule the paper claims."""
    vals = list(per_model_vals.values())
    if any(v is None for v in vals):
        return None
    return vals[0] if all(v == vals[0] for v in vals) else None


def _majority_with_drop_ties(vals: List[Any]) -> Optional[Any]:
    """Helper: 0→None, 1→that value, 2+→majority with drop-ties."""
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    c = Counter(vals)
    top_val, top_count = c.most_common(1)[0]
    n_tied_at_top = sum(1 for v, n in c.items() if n == top_count)
    if top_count >= 2 and n_tied_at_top == 1:
        return top_val
    return None


def _ensemble_pain_aux(per_model_pain: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Per-row pain-auxiliary ensemble. Mirrors _ensemble_demo's union-with-
    drop-ties philosophy, generalised by field type:

      * is_mental (binary), severity (categorical): majority-with-drop-ties.
      * location, pain_type (lists): per-item majority vote across the models
        that extracted *any* item. An item is kept iff at least 2 of the
        extracting models list it (after lowercase/strip normalisation). If
        only one model extracted anything for the field, trust its list.
        Mirrors the demographic rule "1 model → use, 2+ → require majority".

    Rationale: free-text auxiliary fields are noisier than demographics, so
    single-model items (which include most hallucinated locations) are
    dropped when other models had a chance to confirm and did not.
    """
    out: Dict[str, Any] = {}

    # Binary / categorical: majority-with-drop-ties
    for f in ("is_mental", "severity"):
        vals = [m[f] for m in per_model_pain.values() if m[f] is not None]
        out[f] = _majority_with_drop_ties(vals)

    # Lists: per-item vote
    for f in ("location", "pain_type"):
        per_lists = [m[f] for m in per_model_pain.values() if m[f] is not None]
        if not per_lists:
            out[f] = None
            continue
        if len(per_lists) == 1:
            out[f] = list(per_lists[0])
            continue
        item_counts: Counter = Counter()
        for lst in per_lists:
            for it in set(lst):  # one vote per model, even if model lists item twice
                item_counts[it] += 1
        kept = [it for it, n in item_counts.items() if n >= 2]
        out[f] = kept  # may be [] when ≥2 models extracted but no item agreed
    return out


def _ensemble_demo(per_model: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Per-row demographic ensemble (union with majority tie-break).

    For each field, look at all non-null model extractions:
      * 0 non-null   → None  (no model saw the field; nothing to use)
      * 1 non-null   → that value (single-model extraction; we trust it)
      * 2+ non-null  → most common value (majority); ties broken by first
                       extraction in deterministic model-name order

    Rationale: source text for many fields is sparse (e.g. ethnicity often
    not mentioned), and individual models have language-specific recall
    quirks (e.g. Gemma on Portuguese). A strict all-same rule conflates
    "source has no information" with "models disagree" and discards a
    large amount of valid signal, especially in BRIDGE-Pain. The union
    rule keeps every non-null extraction and uses majority where models
    differ. When ≥2 models extract, agreement is 96--100% on age/gender
    after cross-lingual normalisation, so genuine semantic conflicts are
    rare.
    """
    from collections import Counter
    out: Dict[str, Any] = {}
    for f in _ENSEMBLE_FIELDS:
        vals = [m[f] for m in per_model.values() if m[f] is not None]
        if not vals:
            out[f] = None                        # case 1: no model extracted
        elif len(vals) == 1:
            out[f] = vals[0]                     # case 2: single extractor
        else:
            c = Counter(vals)
            top_val, top_count = c.most_common(1)[0]
            # case 3: at least 2 models extracted the same value → consensus
            # case 4: ties at top_count=1 (2-disagree or 3-distinct) → abstain
            # Avoids the silent "first-listed model wins" bias from Counter
            # iteration order on ties.
            n_tied_at_top = sum(1 for v, n in c.items() if n == top_count)
            if top_count >= 2 and n_tied_at_top == 1:
                out[f] = top_val
            else:
                out[f] = None
    return out


def load_worldmedqa(country: str) -> List[Dict[str, Any]]:
    """Row-aligned records for one WMQA country: 3-model ensemble pain
    (binary core + auxiliary) + 3-model ensemble demographics + physician GT.
    """
    cfg = WMQA_FILES[country]
    gt_rows = _read_csv(cfg["gt"])

    pain_per_model: Dict[str, List[Dict[str, str]]] = {}
    demo_per_model: Dict[str, List[Dict[str, str]]] = {}
    for model_name, model_dir in DEMOGRAPHIC_MODEL_DIRS.items():
        pain_path = model_dir / f"{country}_pain.csv"
        demo_path = model_dir / f"{country}_demographics.csv"
        if pain_path.exists():
            pain_per_model[model_name] = _read_csv(pain_path)
        if demo_path.exists():
            demo_per_model[model_name] = _read_csv(demo_path)

    counts: Dict[str, int] = {f"pain[{m}]": len(rs) for m, rs in pain_per_model.items()}
    counts.update({f"demo[{m}]": len(rs) for m, rs in demo_per_model.items()})
    counts["gt"] = len(gt_rows)
    if len(set(counts.values())) > 1:
        print(f"[{country}] WARNING row counts differ: {counts}")

    records: List[Dict[str, Any]] = []
    n_use = min(
        [len(gt_rows)]
        + [len(r) for r in demo_per_model.values()]
        + [len(r) for r in pain_per_model.values()]
    )
    for i in range(n_use):
        g = gt_rows[i]
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
            "row_idx":            i,
            "language":           cfg["lang"],
            "country":            country,
            # GT (physician-validated; None for UK unlabelled rows)
            "gt_pain_related":    _gt_bool(g.get("pain_related (Y/N)")),
            "gt_pain_presence":   _gt_bool(g.get("pain_presence(Y/N)")),
            # Ensemble pain (strict all-same binary, drop-ties aux)
            "pain_related":       pr_bin,
            "pain_presence":      pp_bin,
            "location":           pain_aux["location"],
            "pain_type":          pain_aux["pain_type"],
            "severity":           pain_aux["severity"],
            "is_mental":          pain_aux["is_mental"],
            # Per-model views (for diagnostics / coverage)
            "pain_per_model":     pain_norm,
            "demo_per_model":     demo_norm,
            # Ensemble demographics
            "age":                ens_demo["age"],
            "age_group":          ens_demo["age_group"],
            "gender":             ens_demo["gender"],
            "ethnicity":          ens_demo["ethnicity"],
            "occupation":         ens_demo["occupation"],
        })
    return records


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def _safe_div(a, b):
    return round(a / b, 4) if b else 0.0


def dim1_pain_prevalence(records: List[Dict]) -> Dict:
    """3-model ensemble pain extraction vs physician GT: prevalence + F1.
    Rows where GT is None (UK unlabelled rows) are excluded so rates and F1
    are over the GT-labelled subset only.
    """
    labelled = [r for r in records
                if r["gt_pain_related"] is not None
                and r["gt_pain_presence"] is not None]
    n_total = len(records)
    n       = len(labelled)
    gt_related   = sum(1 for r in labelled if r["gt_pain_related"])
    gt_presence  = sum(1 for r in labelled if r["gt_pain_presence"])
    ens_related  = sum(1 for r in labelled if r["pain_related"])
    ens_presence = sum(1 for r in labelled if r["pain_presence"])

    def _binary_stats(field_pred, field_gt):
        tp = fp = tn = fn = 0
        for r in labelled:
            p = r[field_pred]; gt = r[field_gt]
            if p is None: continue
            if p and gt:     tp += 1
            elif p and not gt: fp += 1
            elif not p and not gt: tn += 1
            else: fn += 1
        prec = _safe_div(tp, tp + fp)
        rec  = _safe_div(tp, tp + fn)
        f1   = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0
        acc  = _safe_div(tp + tn, tp + fp + tn + fn)
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    return {
        "n_total":              n_total,
        "n":                    n,
        "gt_pain_related_pct":  _safe_div(gt_related,  n),
        "gt_pain_presence_pct": _safe_div(gt_presence, n),
        "pain_related_pct":     _safe_div(ens_related,  n),
        "pain_presence_pct":    _safe_div(ens_presence, n),
        "metrics_pain_related":  _binary_stats("pain_related",  "gt_pain_related"),
        "metrics_pain_presence": _binary_stats("pain_presence", "gt_pain_presence"),
    }


def dim2_coverage(records: List[Dict]) -> Dict:
    """Demographic coverage breakdown.

    ``per_model``: per-model non-null rate (raw extractability of each model)
    ``ensemble``:  rows where ALL models present agree on the field (this is
                   the substrate for the bias analyses in dim3 / dim4)
    """
    n = len(records)
    fields = ("age", "gender", "ethnicity", "occupation")
    # Per-model coverage
    per_model_cov: Dict[str, Dict[str, float]] = {}
    if records:
        for m in records[0]["demo_per_model"]:
            per_model_cov[m] = {
                f: _safe_div(
                    sum(1 for r in records if r["demo_per_model"][m][f] is not None),
                    n,
                ) for f in fields
            }
    # Ensemble coverage on the pain-consensus subset (both pain fields
    # resolved), to share one substrate with the stratified analyses.
    cons = [r for r in records
            if r["pain_presence"] is not None and r["pain_related"] is not None]
    n_cons = len(cons)
    ensemble_cov = {
        f: _safe_div(sum(1 for r in cons if r[f] is not None), n_cons)
        for f in fields
    }
    return {
        "n":         n,
        "per_model": per_model_cov,
        "ensemble":  ensemble_cov,
    }


def dim3_demographic_distribution(records: List[Dict]) -> Dict:
    """Distribution of demographics within (a) all rows and (b) pain-present rows."""
    def _subset_dist(rows):
        return {
            "n":          len(rows),
            "gender":     dict(Counter(r["gender"] for r in rows if r["gender"]).most_common()),
            "age_group":  dict(Counter(r["age_group"] for r in rows if r["age_group"]).most_common()),
            "ethnicity":  dict(Counter(r["ethnicity"] for r in rows if r["ethnicity"]).most_common()),
            "occupation_top": dict(
                Counter(r["occupation"] for r in rows if r["occupation"]).most_common(10)
            ),
            "age_mean":   round(
                sum(r["age"] for r in rows if r["age"] is not None) /
                max(sum(1 for r in rows if r["age"] is not None), 1), 1
            ) if any(r["age"] is not None for r in rows) else None,
        }
    pain_rows = [r for r in records if r["pain_presence"]]
    return {"all":  _subset_dist(records), "pain_present": _subset_dist(pain_rows)}


def pain_aux_summary(records: List[Dict], pain_flag: str = "pain_presence") -> Dict[str, Any]:
    """Summary of the ensembled pain auxiliary fields, conditional on
    `pain_flag` == True (so location/pain_type/severity are meaningful).
    Reusable across WMQA and BRIDGE.
    """
    pain_pos = [r for r in records if r.get(pain_flag) is True]
    n_pp = len(pain_pos)
    is_mental_resolved = sum(1 for r in pain_pos if r.get("is_mental") is not None)
    is_mental_true     = sum(1 for r in pain_pos if r.get("is_mental") is True)
    sev_resolved = [r["severity"] for r in pain_pos if r.get("severity") is not None]
    sev_dist     = dict(Counter(sev_resolved).most_common())
    loc_any  = sum(1 for r in pain_pos
                   if isinstance(r.get("location"), list) and r["location"])
    type_any = sum(1 for r in pain_pos
                   if isinstance(r.get("pain_type"), list) and r["pain_type"])
    loc_tokens: Counter = Counter()
    for r in pain_pos:
        if isinstance(r.get("location"), list):
            loc_tokens.update(r["location"])
    type_tokens: Counter = Counter()
    for r in pain_pos:
        if isinstance(r.get("pain_type"), list):
            type_tokens.update(r["pain_type"])
    return {
        "n_pain_present":         n_pp,
        "is_mental_resolved":     is_mental_resolved,
        "is_mental_true":         is_mental_true,
        "is_mental_true_rate":    is_mental_true / is_mental_resolved if is_mental_resolved else 0.0,
        "severity_resolved":      len(sev_resolved),
        "severity_resolved_rate": len(sev_resolved) / n_pp if n_pp else 0.0,
        "severity_dist":          sev_dist,
        "location_any":           loc_any,
        "location_any_rate":      loc_any / n_pp if n_pp else 0.0,
        "pain_type_any":          type_any,
        "pain_type_any_rate":     type_any / n_pp if n_pp else 0.0,
        "top_locations":          dict(loc_tokens.most_common(15)),
        "top_pain_types":         dict(type_tokens.most_common(15)),
    }


def dim4_pain_by_demographic(records: List[Dict]) -> Dict:
    """Pain-presence rate stratified by each demographic + chi-square test.
    Uses the 3-model ensemble pain_presence (strict all-same on binary).
    """
    out = {}
    for field in ("gender", "age_group", "ethnicity"):
        groups: Dict[str, Tuple[int, int]] = {}  # group → (pain_present_count, total)
        for r in records:
            v = r[field]
            # Table 1 consensus substrate: BOTH pain fields ensemble-resolved.
            if v is None or r["pain_presence"] is None or r["pain_related"] is None:
                continue
            cnt, tot = groups.get(v, (0, 0))
            groups[v] = (cnt + (1 if r["pain_presence"] else 0), tot + 1)
        if len(groups) < 2:
            out[field] = {"status": "insufficient groups", "groups": {}}
            continue
        table = [[c, t - c] for c, t in groups.values()]  # rows=groups, cols=[pain, no_pain]
        chi2, df, p = chi_square(table)
        out[field] = {
            "groups": {g: {"pain_present": c, "total": t, "rate": _safe_div(c, t)}
                       for g, (c, t) in groups.items()},
            "chi2": chi2, "df": df, "p": p,
        }
    return out


def dim5_cross_language(per_country: Dict[str, Dict]) -> Dict:
    """Compare key stats across countries + chi-square on pain rate × language."""
    out = {
        "pain_presence_by_country": {
            c: per_country[c]["dim1"]["pain_presence_pct"] for c in per_country
        },
        "demographic_coverage_by_country_ensemble": {
            c: per_country[c]["dim2"]["ensemble"] for c in per_country
        },
        "demographic_coverage_by_country_per_model": {
            c: per_country[c]["dim2"]["per_model"] for c in per_country
        },
    }

    # Cx2 table: country × (pain_present, not_pain_present)
    table = []
    countries = list(per_country)
    for c in countries:
        m = per_country[c]["dim1"]["metrics_pain_presence"]
        pos = m["tp"] + m["fp"]
        neg = m["tn"] + m["fn"]
        table.append([pos, neg])
    chi2, df, p = chi_square(table)
    out["chi2_country_x_pain"] = {"chi2": chi2, "df": df, "p": p, "countries": countries}

    # Cross-country gender distribution among pain-present rows
    gender_table_keys: List[str] = []
    gender_table: List[List[int]] = []
    for c in countries:
        dist = per_country[c]["dim3"]["pain_present"]["gender"]
        for g in dist:
            if g not in gender_table_keys:
                gender_table_keys.append(g)
    for c in countries:
        dist = per_country[c]["dim3"]["pain_present"]["gender"]
        gender_table.append([dist.get(g, 0) for g in gender_table_keys])
    chi2_g, df_g, p_g = chi_square(gender_table)
    out["chi2_country_x_gender_in_pain"] = {
        "chi2": chi2_g, "df": df_g, "p": p_g,
        "rows": countries, "cols": gender_table_keys, "table": gender_table,
    }
    return out


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _print_country_summary(country: str, res: Dict) -> None:
    n = res["dim1"]["n"]
    print(f"\n{'='*70}")
    print(f"  {country}  (n={n})")
    print(f"{'='*70}")

    # Dim 1
    d1 = res["dim1"]
    mr  = d1["metrics_pain_related"]
    mp  = d1["metrics_pain_presence"]
    print(f"\n  [Dim 1] Pain prevalence — 3-model ensemble vs physician GT")
    print(f"    pain_related   GT={d1['gt_pain_related_pct']:.1%}   ENS={d1['pain_related_pct']:.1%}"
          f"   F1={mr['f1']:.3f}  (P={mr['precision']:.3f} R={mr['recall']:.3f})")
    print(f"    pain_presence  GT={d1['gt_pain_presence_pct']:.1%}   ENS={d1['pain_presence_pct']:.1%}"
          f"   F1={mp['f1']:.3f}  (P={mp['precision']:.3f} R={mp['recall']:.3f})")

    # Dim 2 — per-model + ensemble coverage
    d2 = res["dim2"]
    print(f"\n  [Dim 2] Demographic coverage")
    print(f"    {'field':<11} " + " ".join(f"{m:>14}" for m in d2["per_model"]) + f"  {'ens-agree':>10}")
    for f in ("age", "gender", "ethnicity", "occupation"):
        per_m = " ".join(f"{d2['per_model'][m][f]:>13.1%} " for m in d2["per_model"])
        print(f"    {f:<11} {per_m}  {d2['ensemble'][f]:>9.1%}")

    # Dim 3
    d3 = res["dim3"]
    print(f"\n  [Dim 3] Demographic distribution — all rows vs pain-present subset")
    for subset in ("all", "pain_present"):
        s = d3[subset]
        print(f"    [{subset}] n={s['n']}  age_mean={s['age_mean']}")
        if s["gender"]:    print(f"       gender:    {s['gender']}")
        if s["age_group"]: print(f"       age_group: {s['age_group']}")
        if s["ethnicity"]: print(f"       ethnicity: {s['ethnicity']}")

    # Dim 4
    d4 = res["dim4"]
    print(f"\n  [Dim 4] Pain-presence rate × demographic   (chi-square test)")
    for field, info in d4.items():
        if "groups" not in info or not info["groups"]:
            print(f"    {field:<11} {info.get('status', 'no data')}")
            continue
        parts = [f"{g}={d['rate']:.1%} ({d['pain_present']}/{d['total']})"
                 for g, d in info["groups"].items()]
        print(f"    {field:<11} χ²={info['chi2']:>6.2f}  df={info['df']}  p={info['p']:<8}  {' | '.join(parts)}")


def _print_cross_language(d5: Dict) -> None:
    print(f"\n{'='*70}")
    print(f"  [Dim 5] Cross-language comparison")
    print(f"{'='*70}")
    print("\n  Pain-presence rate (DeepSeek-extracted):")
    for c, r in d5["pain_presence_by_country"].items():
        print(f"    {c:<8} {r:.1%}")

    cov = d5["demographic_coverage_by_country_ensemble"]
    print("\n  Demographic ensemble coverage (all-3-models-agree %):")
    print(f"    {'country':<8} {'age':>8} {'gender':>10} {'ethnic.':>10} {'occup.':>10}")
    for c, cv in cov.items():
        print(f"    {c:<8} {cv['age']:>8.1%} {cv['gender']:>10.1%} {cv['ethnicity']:>10.1%} {cv['occupation']:>10.1%}")

    g = d5["chi2_country_x_pain"]
    print(f"\n  Country × pain-presence test:  χ²={g['chi2']}  df={g['df']}  p={g['p']}")
    gx = d5["chi2_country_x_gender_in_pain"]
    print(f"  Country × gender (in pain cases): χ²={gx['chi2']}  df={gx['df']}  p={gx['p']}")
    print(f"     rows={gx['rows']}  cols={gx['cols']}")
    for c, row in zip(gx["rows"], gx["table"]):
        print(f"     {c:<8} {row}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_all(out_dir: Optional[Path]) -> Dict[str, Any]:
    per_country: Dict[str, Dict] = {}
    for country in WMQA_FILES:
        records = load_worldmedqa(country)
        per_country[country] = {
            "dim1":     dim1_pain_prevalence(records),
            "dim2":     dim2_coverage(records),
            "dim3":     dim3_demographic_distribution(records),
            "dim4":     dim4_pain_by_demographic(records),
            "pain_aux": pain_aux_summary(records),
        }
        _print_country_summary(country, per_country[country])

    cross = dim5_cross_language(per_country)
    _print_cross_language(cross)

    summary = {
        "source":         "3-model open-weight ensemble (Gemma4-E4B + Qwen3-8B-Thinking + DeepSeek-R1-Distill-Qwen-32B)",
        "datasets":       ["WorldMedQA-Pain (Brazil/Japan/Spain/UK)"],
        "worldmedqa":     per_country,
        "cross_language": cross,
    }

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "paint_analysis.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  → analysis JSON saved: {out_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="PAINT WMQA bias analysis (3-model ensemble).")
    parser.add_argument("--output", "-o", help="Output directory for analysis JSON")
    args = parser.parse_args()
    out_dir = Path(args.output) if args.output else None
    run_all(out_dir)


if __name__ == "__main__":
    main()
