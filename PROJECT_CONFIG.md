# FairSkin AI dataset project configuration

Target classes:
- acne
- melasma
- vitiligo
- psoriasis
- ringworm

Secondary task:
- acne severity: low/high, only with appropriate clinical annotation

Skin-tone analysis:
- Fitzpatrick I–VI, plus unknown
- preserve annotation source and uncertainty

Evaluation:
- patient/group-aware split where possible
- untouched test set
- disease-by-skin-type metrics
- baseline vs fairness interventions
- out-of-distribution and image-quality rejection
- separate external-source evaluation when feasible

Do not set a minimum sample count as if it guarantees statistical adequacy. Determine sample needs from the intended performance precision, prevalence, subgroup coverage and study design.
