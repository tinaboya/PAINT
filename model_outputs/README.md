# model_outputs/

Consolidated LLM extraction outputs for MedScope. One folder per model,
uniform file naming.

## Layout

```
model_outputs/<model>/<Dataset>_<task>.csv
```

- `<Dataset>` ∈ {Brazil, Japan, Spain, UK}  (WorldMedQA partitions)
- `<task>`    ∈ {pain, demographics}

## Models

| Folder | Role | pain | demographics |
|--------|------|------|--------------|
| `deepseek-r1-qwen-32b/` | ensemble member (reasoning-tuned, 32B) | Brazil/Japan/Spain/UK | Brazil/Japan/Spain/UK |
| `gemma4-e4b/`           | ensemble member (4B)                   | Brazil/Japan/Spain/UK | Brazil/Japan/Spain/UK |
| `qwen3-8b/`             | ensemble member (8B)                   | Brazil/Japan/Spain/UK | Brazil/Japan/Spain/UK |
| `gpt-5/`                | closed-source reference (not in ensemble) | Brazil/Japan/Spain/UK | — |
| `llama-3.1-8b/`         | un-selected baseline (low F1)          | Brazil/Japan/Spain | — |
| `medgemma-1.5/`         | un-selected baseline (low F1)          | Brazil/Japan/Spain | — |

LLaMA-3.1-8B and MedGemma-1.5 were not run on the UK partition.
Demographic extraction exists only for the three ensemble members.

## Schema

Pain CSVs carry a uniform header: `Question_ID, ok, pain_related,
pain_presence, location, pain_type, severity`. (`ok` flags a successfully
parsed extraction.)

Demographic CSVs carry: `Question_ID`/`row_index, age, gender,
ethnicity, occupation`.

The index column follows a different scheme per file (0-based row position
for Gemma and for the Qwen pain files; the WorldMedQA source `Question_ID`,
which repeats across exam files, for DeepSeek and the Qwen demographic
files). Do not join on it: all files of a partition are in the same row
order, and that position is the `row_index` of `data/WorldMedQA-Pain/`.


## Consumers

- `src/paint/build_wmqa_labels.py` — re-derives every `ensemble_*` column of
  `data/WorldMedQA-Pain/` from these per-model outputs (strict all-same rule
  for the two pain fields, 2-of-3 majority for the other fields).

## Provenance

The native-language channel was used throughout (bz_pt, jp_jp, es_es, uk_en).
The three ensemble members are `deepseek-r1-qwen-32b`, `gemma4-e4b`, and
`qwen3-8b`; `gpt-5` is a closed-source reference and `llama-3.1-8b`,
`medgemma-1.5` are un-selected baselines.
