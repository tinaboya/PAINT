# Extraction prompts

The two chain-of-thought prompt templates used for every model run in the
paper. Each file holds four language-specific variants
(`SYSTEM_PROMPT_EN`, `SYSTEM_PROMPT_JP`, `SYSTEM_PROMPT_ES`,
`SYSTEM_PROMPT_BZ`) — each model is given the prompt in the native
language of the corpus it is run on.

| File | Object extracted | Fields |
|------|------------------|--------|
| `pain-extraction-system-prompt-v2.txt` | pain | `pain_related, pain_presence, location, pain_type, severity, is_mental, context` |
| `demographic-extraction-system-prompt.txt` | demographic | `age, gender, ethnicity, occupation` |

Each template contains a single `{text}` placeholder, replaced with the
clinical text at inference time. The English variant of each prompt is
also reproduced verbatim in Appendix A of the paper
(`paper/paint.tex`).

Earlier/abandoned prompt drafts (`baseline.txt`, `cot.txt`, and earlier
pain-prompt versions) are kept in `archive/legacy_prompts/` and are not
used by the paper pipeline.
