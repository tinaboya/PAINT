# Completed clinician annotation sheets

One sheet per (partition, annotator), plus a `_master` sheet per partition that
merges both annotators. Every sheet is the file the clinician returned, with the
`input_text` column removed; no other cell was altered. Rows are keyed by
`row_index` / `orig_id`, which are sampling indices into the upstream corpus,
not patient identifiers.

| Partition | Upstream licence | Released here |
|---|---|---|
| WorldMedQA UK, Spain, Japan, Brazil | public | yes |
| BRIDGE Cantemist | public (Zenodo) | yes |
| BRIDGE IFMIR | public (figshare) | yes |
| BRIDGE MIMIC-III, MEDIQA | PhysioNet credentialed | no — available to credentialed users on request |
| BRIDGE Brateca | PhysioNet credentialed | no — available to credentialed users on request |

Column layout per field: `<field>__consensus` (ensemble label, pre-filled),
`<field>__correct(TRUE/FALSE)` (annotator judgement),
`<field>__true_value_if_false` (correction, supplied when FALSE). A FALSE with
the correction left blank means "the text does not state this value".

Reproduce the paper's validation tables:

    python3 src/paint/clinician_validation.py
    python3 src/paint/inter_annotator_agreement.py
    python3 src/paint/kappa_transparency.py
