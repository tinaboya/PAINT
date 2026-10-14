<p align="center">
  <img src="assets/paint-logo.png" alt="PAINT" width="380">
</p>

<h1 align="center">A Framework for Multilingual Pain Representation<br>from Medical Licensing Exams to Real-World Clinical Texts</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2610.18269"><img src="https://img.shields.io/badge/arXiv-2610.18269-b31b1b" alt="arXiv"></a>
  <a href="https://github.com/tinaboya/PAINT/actions/workflows/check.yml"><img src="https://github.com/tinaboya/PAINT/actions/workflows/check.yml/badge.svg" alt="check"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/languages-EN%20%C2%B7%20ES%20%C2%B7%20PT%20%C2%B7%20JA-orange" alt="Languages">
  <img src="https://img.shields.io/badge/data-WorldMedQA--Pain%20%C2%B7%20BRIDGE--Pain-E4572E" alt="Datasets">
</p>

<p align="center">
  <b>PAINT</b> extracts structured <b>pain</b> and <b>demographic</b> attributes from medical-licensing
  exams (WorldMedQA) and real clinical text (BRIDGE) across four languages, using a strict
  three-model open-weight ensemble to study how pain is framed across languages and care settings.
</p>

<p align="center">
  Paper: <a href="https://arxiv.org/abs/2610.18269">arXiv:2610.18269</a> ·
  Accepted at the 6th Workshop on Multilingual Representation Learning (MRL 2026), EMNLP 2026, Budapest
</p>

<p align="center"><img src="assets/overview.png" alt="PAINT overview" width="760"></p>
<p align="center"><i>Medical licensing examinations and clinical text in four languages, annotated with an ensemble of open-weight language models and validated by clinicians.</i></p>

---

## News

- **2026-10-14** — WorldMedQA-Pain release with ensemble labels for every annotated field (`data/WorldMedQA-Pain/`).
- **2026-10-13** — Preprint on arXiv: [2610.18269](https://arxiv.org/abs/2610.18269).
- **2026-09** — Paper accepted at the 6th Workshop on Multilingual Representation Learning (MRL 2026), EMNLP 2026, Budapest.
- **Coming** — clinician gold labels for all WorldMedQA-Pain items.

## Why PAINT?

Pain is one of the most commonly documented symptoms, yet annotated pain resources are
overwhelmingly **English-only** and **single-institution**. PAINT addresses three gaps:

- **Multilingual** — English, Spanish, Portuguese, and Japanese, in native text.
- **Two document types** — national licensing exams (*WorldMedQA-Pain*) paired with real clinical notes (*BRIDGE-Pain*).
- **Reproducible labels** — an open-weight three-model ensemble, validated against physician annotation, replaces closed-model or manual labelling at scale.

## Datasets

| Corpus | Document type | Languages | Instances | In this release |
|---|---|---|:---:|:---:|
| **WorldMedQA-Pain** | Licensing exams (public) | EN · ES · PT · JA | 2,963 | ✅ with ensemble labels for all annotated fields |
| **BRIDGE-Pain** | Real clinical notes | EN · ES · PT · JA | 17,795 | ↗ from source corpora (see below) |

**WorldMedQA-Pain partitions:** UK PLAB (en, 268) · Spain MIR (es, 1,049) · Japan NLE (ja, 742) · Brazil REVALIDA (pt, 904).

> **What ships here.** The public **WorldMedQA-Pain** items with **ensemble** labels for every
> annotated field (pain relevance, pain presence, anatomical location, pain descriptors, severity,
> age, gender, ethnicity, occupation; `data/WorldMedQA-Pain/`, schema below) and the per-model
> extraction outputs (`model_outputs/`), so the labels are reproducible from the open-weight models alone. The **clinician validation sheets**
> for the publicly licensed corpora (the four WorldMedQA-Pain partitions, Cantemist, IFMIR) are in
> `annotation/completed_sheets/`, with the source-text column removed; sheets for MIMIC-III, MEDIQA
> and Brateca are available to credentialed users on request. Clinician gold labels for **all**
> WorldMedQA-Pain items are held for a subsequent version.
>
> **How to obtain BRIDGE-Pain.** The **BRIDGE-Pain** clinical corpora (MEDIQA, MIMIC-III,
> Cantemist, Brateca, IFMIR) are all **available** — we do not re-host them here because each
> is governed by its own data-use agreement. The open-access corpora are distributed through the
> [**BRIDGE-Open**](https://huggingface.co/datasets/YLab-Open/BRIDGE-Open) collection; the
> controlled-access ones are obtained from their providers (e.g. MIMIC-III under the PhysioNet
> Credentialed Health Data License). Once obtained, regenerate the BRIDGE-Pain labels with this
> pipeline; only aggregate statistics appear in the paper.

### WorldMedQA-Pain file schema

One CSV per partition (`UK`, `Spain`, `Japan`, `Brazil`), one row per exam item, rows in the
order used throughout the repository.

| Column | Content |
|---|---|
| `row_index` | 0-based row position; the join key to `model_outputs/` and to `annotation/completed_sheets/` |
| `Question_ID`, `file_name` | identifiers from the WorldMedQA source (`Question_ID` repeats across exam files; the pair is unique) |
| `Question` | exam item text in the source language, the only input given to the models |
| `Question_EN` | English translation from WorldMedQA (for UK, identical to `Question`) |
| `ensemble_pain_related`, `ensemble_pain_presence` | strict all-same rule: `True` / `False`, or `abstain` when the three models disagree |
| `ensemble_location`, `ensemble_pain_type` | lists merged item by item (2-of-3), items separated by `; `, in the source language |
| `ensemble_severity` | `mild` / `moderate` / `severe` by 2-of-3 majority |
| `ensemble_age`, `ensemble_gender`, `ensemble_ethnicity`, `ensemble_occupation` | 2-of-3 majority over non-null predictions; age in years, gender `male` / `female`, ethnicity and occupation free text |

For the majority fields an empty cell means that no model extracted the field and `abstain` means
that the models disagreed without a majority; both count as missing in the paper's coverage
figures. Pain relevance, pain presence, age, gender and occupation are the fields covered by the
clinician validation reported in the paper; location, descriptors, severity and ethnicity are
ensemble outputs that were not clinician-validated, and ethnicity categories are not comparable
across countries. The unused WorldMedQA columns (answer options, correct answer, specialty, year)
are not carried; they are available from the WorldMedQA source.

### Example record

`data/WorldMedQA-Pain/Brazil.csv`, `row_index` 0 (REVALIDA 2011, item 1):

| Field | Value |
|---|---|
| `Question` | *Mulher com 54 anos de idade, Índice de Massa Corpórea (IMC) = 32,6, portadora de diabetes tipo 2, … comparece à Unidade Básica de Saúde com queixa de dor, do tipo cólica, em hipocôndrio direito, que se irradia para o ombro direito e piora após ingestão de alimentos, especialmente gordurosos. …* |
| `Question_EN` | *A 54-year-old woman, with a Body Mass Index (BMI) of 32.6, with type 2 diabetes, … complaining of cramping pain in the right hypochondrium, which radiates to the right shoulder and worsens after eating, especially fatty foods. …* |
| `ensemble_pain_related` / `ensemble_pain_presence` | `True` / `True` |
| `ensemble_location` | `hipocôndrio direito; ombro direito` |
| `ensemble_pain_type` | `cólica` |
| `ensemble_severity` | empty (no model extracted a severity) |
| `ensemble_age` / `ensemble_gender` | `54` / `female` |
| `ensemble_ethnicity` / `ensemble_occupation` | empty |

### Statistics per partition

Printed by `python3 src/paint/build_wmqa_labels.py`; they match Tables 6 and 7 of the paper up to rounding.

| Partition | Items | Both pain fields resolved | Pain presence | Pain location | Age | Gender | Ethnicity | Occupation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UK PLAB (en) | 268 | 244 (91.0%) | 90 (33.6%) | 88 (32.8%) | 95.5% | 98.4% | 0.8% | 2.0% |
| Spain MIR (es) | 1,049 | 969 (92.4%) | 136 (13.0%) | 124 (11.8%) | 44.3% | 42.6% | 0.4% | 2.0% |
| Japan NLE (ja) | 742 | 689 (92.9%) | 112 (15.1%) | 109 (14.7%) | 49.8% | 49.9% | 0.6% | 5.2% |
| Brazil REVALIDA (pt) | 904 | 804 (88.9%) | 271 (30.0%) | 251 (27.8%) | 83.1% | 84.6% | 1.9% | 9.0% |

Pain presence and pain location are shares of all items (location: pain-presence-positive items
with a non-empty location). The demographic columns are coverage within the items where both pain
fields are resolved. These distributions characterise each examination corpus and are not
prevalence estimates.

## Quickstart

### What runs on a fresh clone

| Command | Needs | Produces |
|---|---|---|
| `python3 src/paint/build_wmqa_labels.py [--check]` | nothing beyond the repository | `data/WorldMedQA-Pain/`, or a zero-difference check |
| `python3 src/paint/clinician_validation.py` | nothing | the clinician validation tables, from `annotation/completed_sheets/` |
| `python3 src/paint/inter_annotator_agreement.py` | nothing | Cohen's kappa and % agreement on the double-annotated overlaps |
| `python3 src/paint/kappa_transparency.py` | nothing | the kappa interpretation note |
| `python3 -m unittest discover -s tests` | nothing | the unit tests (also run by CI) |
| `csv_to_eval_json.py`, `extraction_evaluation.py`, `ensemble.py`, `analysis.py` | the clinician gold labels in `data/raw/WorldMedQA/`, held for a subsequent version | per-model F1, the ensemble comparison, the analysis tables |
| `PAINT-open-source-models/run_inference*.py` | a CUDA GPU with vLLM, and your own copy of the BRIDGE corpora | `model_outputs/` |

**Rebuild the WorldMedQA-Pain labels** (no dependencies, standard library only):

```bash
python src/paint/build_wmqa_labels.py
```

This reads the three ensemble members' per-model outputs and re-derives every `ensemble_*`
column (2,963 items; 89–93% resolved on both pain fields, the remainder marked `abstain`),
writing `data/WorldMedQA-Pain/`. `--check` compares instead of writing and must report 0 differing
cells.

**Run inference from scratch** (needs a CUDA GPU), from `PAINT-open-source-models/`:

```bash
pip install -r requirements.txt
python run_inference.py                                      # WorldMedQA, 5 open-weight models
python run_inference_bridge.py --input <your BRIDGE export .json>      # BRIDGE, not included here
python parse_bridge.py                                       # → model_outputs/
```

**Evaluation & analysis** (no dependencies), from the repo root. The first four commands score
against the clinician-annotated pain labels in `data/raw/WorldMedQA/`, which are held for a
subsequent version; the clinician-validation commands run from the released sheets as-is:

```bash
python3 src/paint/csv_to_eval_json.py      --output outputs/runs/
python3 src/paint/extraction_evaluation.py --force outputs/runs/llm_*.json
python3 src/paint/ensemble.py              --all --output outputs/ensemble/
python3 src/paint/analysis.py              --output outputs/analysis/
python3 src/paint/clinician_validation.py                   # validation tables from annotation/completed_sheets/
python3 src/paint/inter_annotator_agreement.py              # Cohen's kappa / % agreement on the shared records
python3 src/paint/kappa_transparency.py
```

See [`REPRODUCE.md`](REPRODUCE.md) for which command produces which paper number. The GPT-5
reference reported in the paper was produced via a hosted API.

## Models

The ensemble combines three open-weight models distinct in scale, developer, and training recipe;
a pain label is kept only when **all three agree** (strict all-same); otherwise the item is marked
`abstain`.

| Model | Role |
|---|---|
| Gemma-4-E4B (4B) | ensemble member |
| Qwen3-8B (8B) | ensemble member |
| DeepSeek-R1-Distill-Qwen-32B (32B) | ensemble member |
| GPT-5 | closed-source reference (upper bound) |
| LLaMA-3.1-8B · MedGemma-4B | un-selected baselines |

**Ensemble rules** (as in the paper). The strict all-same rule applies to the two pain fields,
pain relevance and pain presence. Demographic fields (age, gender, ethnicity, occupation) use a
**2-of-3 majority over non-null predictions**: a value is kept when at least two models agree, a
value produced by a single model is kept when no other model contradicts it, and the field is left
missing on an unresolved disagreement. Anatomical location and pain descriptors are lists merged
**item by item under the same 2-of-3 rule**; if only one model returns a list, that list is kept.
Age is normalised to years and gender to male/female with a cross-lingual dictionary; ethnicity
and occupation stay free text. All of this is implemented in `src/paint/analysis.py`
(`_ensemble_demo`, `_ensemble_pain_aux`).

## Use and limitations

**Intended use.** PAINT supports multilingual clinical NLP research, information extraction, and
fairness auditing. It is not intended for clinical diagnosis, treatment recommendation, or
automated triage.

**What the labels are.** Pain as documented by clinicians and examination writers, not as reported
by patients; no source corpus carries numeric pain scores. Demographic values are model
extractions aggregated by majority, not verified facts about a person. Location, descriptors,
severity and ethnicity were not clinician-validated, and ethnicity categories are not comparable
across countries. Field definitions and the supporting literature are in
[`pain_definition.md`](pain_definition.md); the validation protocol is in
[`annotation/README.md`](annotation/README.md).

**Comparability.** Corpus purpose, language, country and healthcare system co-vary across
partitions, so label distributions characterise each corpus rather than estimate prevalence.
Differences between partitions are corpus effects, not claims about pain experience across
cultures. The English partitions are not country-matched (UK PLAB versus US clinical corpora),
and both Portuguese partitions are Brazilian.

**Data governance.** WorldMedQA contains no patient-identifiable information. The annotation
study was granted IRB exemption by MIT, and the ten annotators (nine physicians and one
pharmacist) are co-authors. No clinical source text is redistributed; the released clinician
sheets have the source-text column removed. The MIT license covers the code and the
WorldMedQA-Pain labels, not the upstream clinical corpora, which stay under their own agreements.

## Repository layout

```
data/WorldMedQA-Pain/     WorldMedQA-Pain items with ensemble labels for all annotated fields (4 partitions, n=2,963)
model_outputs/            per-model pain + demographic extractions (6 models)
src/paint/                evaluation, ensemble, and analysis (standard library only)
  build_wmqa_labels.py    rebuilds data/WorldMedQA-Pain/ from model_outputs/ (all fields)
  analysis.py             2-of-3 aggregation of demographics, location, descriptors
  clinician_validation.py validation and agreement tables from the completed sheets
PAINT-open-source-models/ open-weight model inference (vLLM), WorldMedQA + BRIDGE
prompts/                  pain + demographic extraction prompts (EN/ES/PT/JA)
annotation/               scripts that build the clinician annotation sheets
  completed_sheets/       returned sheets, public corpora only (source text removed)
tests/                    unit tests for the label builder (run by .github/workflows/check.yml)
```

**Documentation:** [`REPRODUCE.md`](REPRODUCE.md) (which command produces which paper number),
[`pain_definition.md`](pain_definition.md) (field definitions and literature),
[`annotation/README.md`](annotation/README.md) (clinician validation protocol),
[`prompts/README.md`](prompts/README.md), [`model_outputs/README.md`](model_outputs/README.md),
[`data/WorldMedQA-Pain/README.md`](data/WorldMedQA-Pain/README.md) (dataset card).

## Contributing

Issues and pull requests are welcome. To add a language or a corpus: translate the two prompts
(see [`prompts/README.md`](prompts/README.md)) keeping the JSON schema, extend the pain lexicon in
[`pain_definition.md`](pain_definition.md), run the three ensemble members with
`PAINT-open-source-models/`, and validate a stratified sample with a clinician using
`annotation/build_annotation_sheets.py`. Contributions must respect the data-use agreement of the
source corpus; do not submit clinical source text to this repository.

## Contact

Boya Zhang (University of Geneva), Yuanchao Ma (McGill University) and Jiageng Wu (Harvard
Medical School) are the co-first authors; Douglas Teodoro (University of Geneva) and Leo Anthony
Celi (MIT) supervised the project. Please use GitHub issues for questions about the data and code.

## Citation

If you use PAINT, its labels, or the pipeline, please cite:

```bibtex
@misc{zhang2026paint,
  title         = {{PAINT}: A Framework for Multilingual Pain Representation from Medical Licensing Exams to Real-World Clinical Texts},
  author        = {Zhang, Boya and Ma, Yuanchao and Wu, Jiageng and Chaisutyakorn, Krittaphas and Gameiro, Rodrigo and Han, Ahram and Huang, Chien-Yu and Jaiswal, Nikhil and Leite, Marianna and Mora-Mau, Miriam and Pineda-Moreno, Jaime and Placino, Siena and Yamamoto, Ryohei and Jia, Yugang and Yang, Jie and Celi, Leo Anthony and Teodoro, Douglas},
  year          = {2026},
  eprint        = {2610.18269},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2610.18269},
  note          = {Accepted at the 6th Workshop on Multilingual Representation Learning (MRL 2026), EMNLP 2026, Budapest}
}
```

Please also cite the source corpora you use: WorldMedQA for the exam partitions and, for
BRIDGE-Pain, BRIDGE together with the underlying clinical corpora.

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). WorldMedQA-Pain labels are
released with the corpus. The BRIDGE-Pain corpora are available from their original providers
(open-access ones via [BRIDGE-Open](https://huggingface.co/datasets/YLab-Open/BRIDGE-Open))
under their respective data-use agreements, and are not re-hosted here.
