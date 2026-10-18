# FairSkin AI — Dataset Expansion Starter Kit

This starter kit is designed to expand your existing Fitzpatrick17k metadata into a traceable, licensed, clinically reviewed research dataset.

## Included
- Your original CSV copied into `data/manifests/`
- `scripts/audit_sources.py`: audit label × Fitzpatrick type distributions
- `scripts/download_approved.py`: download only rows explicitly marked `license_status=verified`
- `data/manifests/source_registry.csv`: starting registry of candidate sources and licence checks
- `data/manifests/approved_downloads_template.csv`: source/provenance manifest template
- `data/annotations/ANNOTATION_GUIDE.md`: label and annotation policy
- `PROJECT_CONFIG.md`: target classes and evaluation requirements

## Run the audit
Install pandas:
```
pip install pandas
```
Then:
```
python scripts/audit_sources.py data/manifests/fitzpatrick17k.csv --out reports/fitzpatrick17k
```

## Download guard
Install requests and pandas:
```
pip install pandas requests
```
Fill `data/manifests/approved_downloads_template.csv` only with sources/URLs you have verified you are permitted to use. Set `license_status` to `verified` only after you have documented permission and put the source/permission reference in the row. Then:
```
python scripts/download_approved.py data/manifests/approved_downloads_template.csv --out data/raw
```
The downloader deliberately skips unverified rows.

## Important
This does not automatically certify legal rights, download a huge dataset, or guarantee medical accuracy. Review the actual terms for each image source and your planned training, publication, and mobile distribution. Do not fabricate clinical images or labels. Synthetic images, if studied, must be separately identified and never counted as real clinical evidence.

Before training, inspect the actual label coverage, image integrity, duplicates, source overlap, patient grouping, skin-type annotation provenance, and dataset licence.
