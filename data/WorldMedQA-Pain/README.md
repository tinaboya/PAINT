---
pretty_name: WorldMedQA-Pain
language:
  - en
  - es
  - ja
  - pt
license: mit
task_categories:
  - text-classification
tags:
  - medical
  - clinical-nlp
  - pain
  - multilingual
  - licensing-exams
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: uk
        path: UK.csv
      - split: spain
        path: Spain.csv
      - split: japan
        path: Japan.csv
      - split: brazil
        path: Brazil.csv
---

# WorldMedQA-Pain

Pain and demographic labels for 2,963 medical licensing examination items in four languages,
the exam half of PAINT ([paper](https://arxiv.org/abs/2610.18269), [code](https://github.com/tinaboya/PAINT)).
Labels come from a three-model open-weight ensemble (Gemma-4-E4B, Qwen3-8B,
DeepSeek-R1-Distill-Qwen-32B) and were validated by clinicians in each language.

## Files

| File | Partition | Language | Items |
|---|---|---|---:|
| `UK.csv` | UK PLAB | en | 268 |
| `Spain.csv` | Spain MIR | es | 1,049 |
| `Japan.csv` | Japan NLE | ja | 742 |
| `Brazil.csv` | Brazil REVALIDA | pt | 904 |

```python
from datasets import load_dataset
ds = load_dataset("tinaboya/WorldMedQA-Pain")   # splits: uk, spain, japan, brazil
```

One row per exam item. Items and English translations are from
[WorldMedQA](https://github.com/WorldMedQA/V); only the item text was given to the models.

## Fields

| Column | Content |
|---|---|
| `row_index` | 0-based row position; the join key to the per-model outputs and the clinician sheets in the code repository |
| `Question_ID`, `file_name` | identifiers from the WorldMedQA source (`Question_ID` repeats across exam files; the pair is unique) |
| `Question` | exam item text in the source language |
| `Question_EN` | English translation (for UK, identical to `Question`) |
| `ensemble_pain_related`, `ensemble_pain_presence` | `True` / `False` when the three models agree, otherwise `abstain` |
| `ensemble_location`, `ensemble_pain_type` | lists merged item by item (kept when at least two models list an item), `; `-separated, source language |
| `ensemble_severity` | `mild` / `moderate` / `severe`, 2-of-3 majority |
| `ensemble_age`, `ensemble_gender`, `ensemble_ethnicity`, `ensemble_occupation` | 2-of-3 majority over non-null predictions; age in years, gender `male` / `female`, ethnicity and occupation free text |

For the majority fields an empty cell means that no model extracted the field and `abstain`
means that the models disagreed without a majority.

## Validation and limitations

Pain relevance, pain presence, age, gender and occupation were checked by clinicians (see the
paper); location, descriptors, severity and ethnicity are ensemble outputs that were not
clinician-validated. Demographic values are model extractions, not verified facts about a
person, and ethnicity categories are not comparable across countries. Label distributions
describe each examination corpus and are not prevalence estimates. The items are public exam
questions and contain no patient-identifiable information.

## License and citation

Labels are released under the MIT license, with the items as distributed by WorldMedQA. Please
cite PAINT (arXiv:2610.18269; BibTeX in the code repository) and WorldMedQA.
