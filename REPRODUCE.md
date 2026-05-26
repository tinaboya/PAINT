# Reproducing the PAINT paper

The open-weight results are reproducible end-to-end: **Stage 0** runs model
inference (needs a CUDA GPU + the `PAINT-open-source-models/requirements.txt`
dependencies), and **Stages 1–4** evaluate and analyse the outputs using only
the Python 3.9+ standard library — no `pip install`.

> The GPT-5 reference and Gemini baseline numbers in the paper were produced
> through hosted APIs; that calling code is not part of this repository. Every
> *open-weight* number (Table 3, §4.3, §5) is reproducible from here.

Run the stages in order from the repository root. Stages 1–4 assume
`model_outputs/` (from Stage 0) and `data/raw/WorldMedQA/` are present; neither
is shipped with this anonymized release (see the README data note).

---

## Stage 0 — Inference (produces `model_outputs/`)

Serves each open-weight model with vLLM and runs the pain + demographic
prompts. Outputs `model_outputs/<model>/<Dataset>_{pain,demographics}.csv`.

```bash
cd PAINT-open-source-models
pip install -r requirements.txt

# WorldMedQA — five open-weight models, all partitions:
python run_inference.py

# BRIDGE — three ensemble models, then parse raw outputs to CSV:
python run_inference_bridge.py --input ../data/raw/BRIDGE.example.all.json
python parse_bridge.py
```

The five models and their HuggingFace tags are in `config.py`. Use
`--model <key>` / `--dataset <name>` / `--dry-run` to scope a run.

---

## Stage 1 — Convert model CSVs to evaluation JSON

```bash
python3 src/paint/csv_to_eval_json.py --output outputs/runs/
```

Reads `model_outputs/<model>/<Dataset>_pain.csv` for all six models, joins
each row to the physician ground truth in `data/raw/WorldMedQA/`, and
writes one `outputs/runs/llm_<model>_..._<Dataset>_original.json` per
(model, dataset). Models without a UK run (GPT-5, LLaMA-3.1, MedGemma)
print a `SKIP` line — expected.

**Output:** ~21 JSON files in `outputs/runs/`.

---

## Stage 2 — Compute F1 (paper Table 3)

```bash
python3 src/paint/extraction_evaluation.py --force outputs/runs/llm_*.json
```

Computes binary F1, macro F1, precision, and recall for `pain_related`
and `pain_presence` on every run JSON and writes the metrics back into
each file. The printed summary table is the source of **Table 3 (§4.2)**.

**Verify:** Gemma4-e4b on Brazil should print `0.8650 / 0.8202` (binary
F1, pain-related / pain-presence). DeepSeek-R1-Qwen-32B on UK should print
`0.9244 / 0.9372`.

---

## Stage 3 — Pain ensemble + gold subset (paper §4.3)

```bash
python3 src/paint/ensemble.py --all --output outputs/ensemble/
```

Runs the strict-majority-vote pain ensemble over the three open-weight
models and extracts the all-same gold subset (rows where all three models
agree on every pain field).

**Verify:** `ALL pain fields` agreement should be
`804/904 (88.9%)` for Brazil, `689/742 (92.9%)` for Japan,
`969/1049 (92.4%)` for Spain.

**Output:** `outputs/ensemble/ensemble_<Dataset>.json` and
`gold_<Dataset>_pain.csv`.

---

## Stage 4 — Demographic ensemble + bias analysis (paper §5)

```bash
python3 src/paint/analysis.py --output outputs/analysis/
```

Loads the three-model demographic ensemble, computes the cross-cultural
bias statistics, and runs the BRIDGE pilot analysis.

**Verify:**
- `Country × pain-presence test: χ²=143.798 df=3 p=<0.001` — paper §5.1.
- Brazil age-group: `χ²=15.93 ... adult=46.2% elderly=30.4% pediatric=30.4%` — paper §5.2.
- Brazil gender: `χ²=1.79 ... female=41.2% male=46.9%` (n.s.) — paper §5.3.
- Demographic coverage table — paper §5.4.

**Output:** `outputs/analysis/paint_analysis_deepseek.json`.

---

## What maps to what

| Paper element | Produced by | Verify against |
|---------------|-------------|----------------|
| Table 1 (corpus stats) | `data/raw/WorldMedQA/*.csv` directly | row counts + GT label rates |
| Table 3 (per-model F1) | Stage 2 | printed summary table |
| §4.3 gold-subset rates | Stage 3 | `ALL pain fields` lines |
| §5.1 Table 4 + χ² | Stage 4 | `Country × pain-presence` + `Dim 1` lines |
| §5.2 age-group table | Stage 4 | per-country `Dim 4 age_group` lines |
| §5.3 gender table | Stage 4 | per-country `Dim 4 gender` lines |
| §5.4 coverage table | Stage 4 | per-country `Dim 2` blocks (`ens-agree` column) |
| §5.5 precision/recall | Stage 4 | `Dim 1` per-country P/R |
| §5.6 BRIDGE pilot | Stage 4 | `BRIDGE pilot` block |

## Notes

- **Ground truth.** `data/raw/WorldMedQA/{Brazil,Japan,Spain,UK_T}.csv`
  carry the physician-validated pain labels and are committed to the repo.
  Brazil/Japan/Spain use `Y/N` labels; UK uses `Yes/No` — the loaders
  handle both.
- **Row alignment.** Model CSVs and ground truth are aligned by row
  position, not by `Question_ID` (the per-model ID schemes differ). All
  files in a partition share the same row count (Brazil 904, Japan 742,
  Spain 1049, UK 268).
- **Demographic ensemble.** Only the three ensemble models
  (Gemma4-e4b, Qwen3-8B, DeepSeek-R1-Qwen-32B) carry demographic CSVs.
  `analysis.py` keeps a field's value only when all three agree on the
  normalised value (see the `GENDER_MAP` / `ETHNICITY_MAP` /
  `parse_age` helpers in `src/paint/analysis.py`).
- **Metrics.** Binary F1 is the positive-class F1 (scikit-learn
  `average='binary'` default); macro F1 averages the positive and
  negative class F1 (`average='macro'`). Both are stored in the run JSON.
- **Paper build.** The paper compiles with **LuaLaTeX** (not pdfLaTeX) so
  that Japanese kanji/kana render natively — see `paper/.latexmkrc`.
  Rebuild with `cd paper && latexmk paint.tex`, or manually:
  `lualatex paint && bibtex paint && lualatex paint && lualatex paint`.
