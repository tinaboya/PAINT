# Pain Annotation: Definitions and Literature Support

## Overview

This document defines the two core annotation fields used in our pain extraction
pipeline — `pain_related` (Pain Mention) and `pain_presence` (Pain Presence) —
and provides supporting literature for each definition.

---

## 1. Pain Mention (`pain_related`)

### Definition
A text is marked as **pain-related (`true`)** if any pain-related term appears
anywhere in the text, **regardless of context** (affirmative, negated, historical,
or current).

This includes:
- Direct synonyms of "pain": `pain`, `ache`, `discomfort`, `soreness`,
  `tenderness`, `hurt`, `agony`, `cramp`, `spasm`, `twinge`, `pang`, etc.
- Morphological derivatives: `painful`, `painless`
- Compound/contextual forms: `painkiller`, `no pain`, `pain-free`

If **none** of the above appear → `pain_related = false`

### Annotation Logic
> Pain Mention is a **lexical-level trigger**: it flags whether pain vocabulary
> is present, without evaluating whether pain actually occurred.

### Literature Support
Kreimeyer et al. (2017) define Pain Mention as:
> "The presence or absence of pain as experienced by the patient. Words such as
> 'aching', 'discomfort', 'tenderness' are all considered pain mentions."

**Reference:**
> Kreimeyer, K., et al. (2017). *Classifying Clinical Notes with Pain Assessment
> using Machine Learning*. AMIA Annual Symposium Proceedings.
> https://pmc.ncbi.nlm.nih.gov/articles/PMC6014866/

---

## 2. Pain Presence (`pain_presence`)

### Definition
**Conditional on `pain_related = true`**, a text is marked as
**pain present (`true`)** if pain is **positively affirmed as having occurred**,
at **any point in time** — current or historical.

`pain_presence = false` **only if** pain is **explicitly negated or denied**,
such as:
- `"no pain"`, `"pain-free"`, `"denies pain"`
- `"painless"`, `"without discomfort"`
- `"pain has resolved"`, `"no longer in pain"`

### Key Distinction
> The determining factor is **affirmation vs. negation**, NOT present vs. past
> tense.
>
> - `"She had chest pain last week"` → `pain_presence = true` ✅
> - `"The procedure was painless"` → `pain_presence = false` ✅
> - `"He was prescribed painkillers for back pain"` → `pain_presence = true` ✅

### Annotation Logic
| Scenario | `pain_presence` |
|---|---|
| Current active pain | `true` |
| Historical / past pain | `true` |
| Reported / recalled pain | `true` |
| Explicit negation (`no pain`, `painless`) | `false` |
| Resolved pain (`pain resolved`, `no longer in pain`) | `false` |

### Literature Support
Lavertu et al. (2023) annotated pain mentions in mental health EHR using a
**relevant vs. negated** framework, explicitly stating:
> "Mentions were marked as **negated** if they referenced **absence** of pain,
> such as 'she was not in pain,' 'no pain reported,' or 'he does not complain of
> headaches.' Correct mentions were labeled as **relevant** if it was the patient
> in question who was experiencing the pain."

Crucially, this framework does **not** distinguish between current and historical
pain — both are classified as **relevant (present)**, consistent with our
annotation guideline.

**Reference:**
> Lavertu, A., et al. (2023). *Development of a Corpus Annotated With Mentions
> of Pain in Mental Health Electronic Health Records*. JMIR Formative Research.
> https://formative.jmir.org/2023/1/e45849

---

## 3. Relationship Between the Two Fields

Text Input
│
▼
Does any pain-related term appear? (lexical check)
│
├── NO → pain_related = false, pain_presence = false (stop)
│
└── YES → pain_related = true
│
▼
Is pain affirmed (current or historical)?
│
├── YES (no negation) → pain_presence = true → proceed to location/type
│
└── NO (explicit negation) → pain_presence = false (stop)

---

## 4. Summary of Key References

| Reference | Relevance |
|---|---|
| Kreimeyer et al. (2017). *Classifying Clinical Notes with Pain Assessment using Machine Learning*. AMIA. | Defines Pain Mention as a lexical-level trigger; synonyms and derivatives all count |
| Lavertu et al. (2023). *Development of a Corpus Annotated With Mentions of Pain in Mental Health EHR*. JMIR Formative Research. | Defines Pain Presence via relevant/negated framework; historical pain = present |
