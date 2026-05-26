# PAINT: A Framework for Multilingual Pain Representation from Medical Licensing Exams to Real-World Clinical Texts

PAINT extracts structured **pain** and **demographic** attributes from
medical-licensing exams (WorldMedQA) and real clinical text (BRIDGE) in four
languages (English, Spanish, Portuguese, Japanese), and uses a three-model
ensemble to study cross-cultural patterns of pain framing.

> **Code-only release.** Datasets and per-model outputs (`data/`,
> `model_outputs/`, `outputs/`) are **not** included — the clinical corpora are
> governed by separate data-use agreements (e.g. MIMIC/PhysioNet). Obtain the
> benchmarks from their original releases, then place WorldMedQA-V ground truth
> under `data/raw/WorldMedQA/` and the BRIDGE sample under `data/raw/`.

## Layout

- `PAINT-open-source-models/` — open-weight model inference (vLLM), WorldMedQA + BRIDGE
- `src/paint/` — evaluation, ensemble, and cross-cultural analysis (standard library only)
- `prompts/` — pain + demographic extraction prompts (EN/ES/PT/JA)
- `annotation/` — scripts that build the human-annotation sheets

## Run

**Inference** (needs a CUDA GPU), from `PAINT-open-source-models/`:

```bash
pip install -r requirements.txt
python run_inference.py                                      # WorldMedQA, 5 models
python run_inference_bridge.py --input ../data/raw/BRIDGE.example.all.json
python parse_bridge.py                                       # → model_outputs/
```

**Evaluation & analysis** (no dependencies), from the repo root:

```bash
python3 src/paint/csv_to_eval_json.py      --output outputs/runs/
python3 src/paint/extraction_evaluation.py --force outputs/runs/llm_*.json
python3 src/paint/ensemble.py              --all --output outputs/ensemble/
python3 src/paint/analysis.py              --output outputs/analysis/
```

See [`REPRODUCE.md`](REPRODUCE.md) for which command produces which paper
number. The GPT-5 and Gemini baselines reported in the paper were produced via
hosted APIs; that calling code is not included.

## License

Code is released under the MIT License (see `LICENSE`). The data is licensed
separately and is not redistributed here.
