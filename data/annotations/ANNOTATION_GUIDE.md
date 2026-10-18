# FairSkin AI annotation guide

Do not label uncertain images by guessing.

## Required fields
- image_id: stable anonymous image ID
- disease_label: one of acne, melasma, vitiligo, psoriasis, ringworm, other/unknown
- disease_label_source: dataset, dermatologist, clinical test, or other documented reference
- skin_type: I–VI or unknown; record whether self-reported, dermatologist-estimated, or another documented method
- acne_severity: low/high only for acne and only under a documented dermatologist-approved rubric
- visible_features: multi-label list based on an explicit annotation guide
- lesion_mask: optional path to a verified mask
- label_confidence: reviewer confidence or unresolved
- reviewer_id: pseudonymous reviewer code
- adjudication_status: unreviewed, agreed, adjudicated, uncertain
- permission_reference: source licence or consent/approval reference

## Rules
- Keep original labels and project mappings separate.
- Preserve unknown and uncertain labels.
- Do not use image colour editing to assign a new Fitzpatrick label.
- For a research collection, obtain ethics/IRB or institutional approval where required and informed consent.
- Avoid face and other identifying information unless necessary and explicitly approved.
