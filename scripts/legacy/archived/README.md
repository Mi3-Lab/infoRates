# Archived scripts

These scripts were archived because a unified data management CLI (`scripts/manage_data.py`) and more general utilities have been introduced to centralize and standardize data preparation workflows.

Archived files:
- `build_ucf101_full_manifest.py` — superseded by `scripts/data_processing/build_manifest_from_clips.py` and `scripts/manage_data.py build-manifest`
- `build_ucf101_official_manifests.py` — superseded by `scripts/manage_data.py build-manifest` and dataset handlers
- `build_ucf101_test_manifest.py` — superseded by `scripts/manage_data.py` and the manifest building utilities

Rationale:
- Avoid duplication, reduce maintenance burden, and provide a single, documented entrypoint for data tasks.
- These files are preserved for provenance and reproducibility; if you want them permanently removed, confirm and we will delete them after an integration test.

If you need help migrating any custom workflow from an archived script to the new CLI, I can assist.