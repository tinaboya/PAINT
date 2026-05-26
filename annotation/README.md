# BRIDGE annotation task

The 5 BRIDGE datasets have **no ground-truth labels**, so we validate our
extractions by hand. For each item you judge whether the model **consensus**
value (the agreement across our 3 models) is correct.

## What to do

Open your CSV (e.g. `sheets/pt_Brateca_annotatorA.csv`) in Excel / Google Sheets.
Read `input_text`, then for each field fill the two blank columns:

- `<field>__correct(TRUE/FALSE)` — `TRUE` if the consensus value is right, else `FALSE`
- `<field>__true_value_if_false` — only when you marked FALSE: write the correct value

Do **not** edit the `__consensus` or `input_text` columns.

### Fields

| field | meaning | true value format when FALSE |
|---|---|---|
| `pain_related` | text mentions pain in any context (incl. denied) | `true` / `false` |
| `pain_presence` | pain is affirmed as actually occurring (now or past) | `true` / `false` |
| `age` | age as written in the text | e.g. `41 anos`, or `null` if none |
| `gender` | gender/sex as written | e.g. `Male`, or `null` if none |
| `occupation` | occupation/employment | e.g. `nurse`, or `null` if none |

An empty `__consensus` cell means the models extracted **nothing** (null). Mark
`TRUE` if the text indeed contains no such information.

A `DISAGREE` consensus means the models split with no majority — always supply
the true value.

## Sampling design

**Rows and consensus values come straight from the paper pipeline**
(`paint.bridge_analysis.load_bridge_task`), so each sheet row is exactly the
gold record the paper reports — no separate re-implementation.

- **Inclusion:** only rows where the pain annotation is identical across all 3
  models — both `pain_related` AND `pain_presence` strict all-same (Table 1's
  consensus substrate). Pain-disagreement rows are excluded.
- **Pain** on those rows = the agreed value.
- **Demographics** = the 3-model union-with-majority ensemble (`_ensemble_demo`)
  after cross-lingual canonicalisation (`masculino`==`Male`, `41 anos`==`41`),
  i.e. the same demographic numbers as the paper's Table 2. A blank consensus
  means the ensemble extracted nothing for that field.

Eligible (pain-agree) rows per dataset: MEDIQA 1402, MIMIC 3993, Cantemist 1091,
IFMIR 4744, Brateca 3411 — all well above 150.

From the eligible rows we sample **150** per dataset (seed-fixed, reproducible):

- **Annotator A** labels rows 1–100, **Annotator B** labels rows 51–150.
- The middle **50** rows are labeled by both → inter-annotator agreement.

`<slug>_master.csv` keeps all 150 rows plus the per-model `__agreement` counts
(e.g. `3/3`, `2/3`) for auditing — annotators don't need it.

## WorldMedQA demographic sheets (`sheets_wmqa/`)

WorldMedQA ships **gold pain labels** but **no demographic labels**, and the
paper reports demographic results — so demographics there need human
validation too. Pain is **not** re-annotated on WorldMedQA (gold already
exists in `data/raw/WorldMedQA_combined.csv`).

- **Lighter than BRIDGE** (50/split, A=rows 1–30, B=rows 21–50, 10 overlap):
  WMQA demographic extraction is near-ceiling because the vignettes state
  age/gender explicitly, so a small sample suffices to confirm it; the real
  demographic-quality signal lives in the BRIDGE EHR sets.
- Fields: `age`, `gender`, `occupation` only (no pain).
- Consensus is over the **3 models that ran demographics on WorldMedQA**
  (deepseek-r1-qwen-32b, gemma4-e4b, qwen3-8b) — the same trio behind the
  BRIDGE consensus. GPT-5 / llama-3.1-8b / medgemma-1.5 have pain only on
  WorldMedQA; if the paper cites their demographics, that run is missing.
- Note: some WorldMedQA items aren't patient vignettes (no person described),
  so an empty consensus there is the correct answer.

## Coverage caveat (demographics)

Demographic precision is only measurable where demographics exist. age/gender
coverage is high in EHR-style BRIDGE sets (MIMIC, Cantemis, Brateca) and in
WorldMedQA Brazil/UK; it is genuinely sparse in chat/incident data (MEDIQA,
IFMIR) and weaker in WorldMedQA Japan/Spain. For the sparse cases we report
coverage + a null spot-check (confirming the model correctly abstains) rather
than forcing underpowered per-dataset precision numbers. occupation is sparse
almost everywhere.

## Regenerate

```bash
python3 annotation/build_annotation_sheets.py   # BRIDGE pain + demographics
python3 annotation/build_wmqa_demo_sheets.py     # WorldMedQA demographics
```

Both builders call the paper's analysis pipeline directly, so the consensus
values are identical to the published tables:
- BRIDGE: `paint.bridge_analysis.load_bridge_task` (pain + demo ensembles
  over `model_outputs/<model>/BRIDGE_<task>_{pain,demographics}.csv`); input
  text and `orig_id` joined by row position from
  `BRIDGE_result/pain.<lang>.<dataset>/<model>/*.json`.
- WorldMedQA: `paint.analysis.load_worldmedqa` (demo ensemble over
  `model_outputs/<model>/{Brazil,Japan,Spain,UK}_demographics.csv`); question
  text joined by row position from `data/raw/WorldMedQA/<split>.csv`.
