# ACCV 2026 Dataset Preparation

This document defines the dataset preparation policy before large-scale training.

## Priority

1. **Something-Something V2**
   - Primary temporal-reasoning dataset.
   - Required for the ACCV version.
   - Official source: Qualcomm developer page.
   - The official page reports 220,847 videos, 168,913 train videos, 24,777 validation videos, 27,157 test videos, 174 labels, 12 FPS, and 19.4 GB total download size.

2. **Diving48**
   - Strong secondary temporal dataset.
   - Recommended because it was designed to reduce static/background bias.
   - Official source: UCSD SVCL RESOUND/Diving48 page.
   - The official page reports about 18k trimmed clips, 48 classes, about 16k train videos, about 2k test videos, and RGB download size around 9.6 GB.

3. **FineGym**
   - Optional stretch dataset.
   - Very relevant for fine-grained temporal recognition, but more operationally complex.
   - Use only after Something-Something V2 and Diving48 are stable.

4. **UCF101 / HMDB51**
   - Supporting datasets.
   - Useful for continuity and smoke tests.

5. **Kinetics-400**
   - Supporting only unless canonical validation is rebuilt.
   - The existing K4TestSet/Kaggle-style protocol should not be central to ACCV claims.

## Local Dataset State

Current local data includes:

- `data/UCF101_data/`
- `data/HMDB51_data/`
- `data/Kinetics400_data/`
- `data/Something_data/`

Something-Something V2 labels are present. The dataset has now been downloaded and extracted under scratch storage, with `data/Something_data/videos` symlinked to the extracted video folder.

Diving48 RGB videos have also been downloaded and extracted under scratch storage. The official UCSD JSON annotation URLs returned `403`/timeouts from this environment, so a usable annotation manifest was generated from the OpenMMLab/PYSKL `diving48_hrnet.pkl` file. This manifest is sufficient for pilot experiments, but official V2 JSONs are still preferred if access becomes available.

## Directory Policy

Raw datasets stay under `data/`.

New ACCV manifests and reports go under:

```text
evaluations/accv2026/manifests/
```

Do not overwrite existing ECCV-era outputs. New experiment outputs go under:

```text
evaluations/accv2026/
```

Large archives and extracted videos are stored under scratch to avoid the `/data` quota:

```text
/scratch/wesleyferreiramaia/infoRates/Something_data/
/scratch/wesleyferreiramaia/infoRates/Diving48_data/
```

The repo-visible dataset paths are symlinks:

```text
data/Something_data/videos -> /scratch/wesleyferreiramaia/infoRates/Something_data/videos_full
data/Something_data/raw_archives -> /scratch/wesleyferreiramaia/infoRates/Something_data/raw_archives
data/Diving48_data/videos -> /scratch/wesleyferreiramaia/infoRates/Diving48_data/videos
data/Diving48_data/raw_archives -> /scratch/wesleyferreiramaia/infoRates/Diving48_data/raw_archives
data/Diving48_data/annotations -> /scratch/wesleyferreiramaia/infoRates/Diving48_data/annotations
```

## Audit Commands

### Something-Something V2

Fast audit without probing video metadata:

```bash
python scripts/accv2026/01_audit_datasets.py something \
  --data-root data/Something_data \
  --output-dir evaluations/accv2026/manifests \
  --splits train validation
```

Audit with metadata probing for all existing videos:

```bash
python scripts/accv2026/01_audit_datasets.py something \
  --data-root data/Something_data \
  --output-dir evaluations/accv2026/manifests \
  --splits train validation \
  --probe
```

Create a class-balanced pilot subset after probing:

```bash
python scripts/accv2026/01_audit_datasets.py something \
  --data-root data/Something_data \
  --output-dir evaluations/accv2026/manifests \
  --splits train validation \
  --probe \
  --samples-per-class 20
```

### UCF101 Smoke-Test Manifest

```bash
python scripts/accv2026/01_audit_datasets.py video-tree \
  --dataset ucf101 \
  --video-root data/UCF101_data/UCF-101 \
  --split all \
  --output-dir evaluations/accv2026/manifests \
  --probe-limit 200 \
  --probe
```

### HMDB51 Smoke-Test Manifest

```bash
python scripts/accv2026/01_audit_datasets.py video-tree \
  --dataset hmdb51 \
  --video-root data/HMDB51_data/videos \
  --split all \
  --output-dir evaluations/accv2026/manifests \
  --probe-limit 200 \
  --probe
```

## Download Notes

### Something-Something V2

The official download is behind the Qualcomm research-use license workflow. Complete the license/download flow in the browser, then place all archive parts under:

```text
data/Something_data/raw_archives/
```

After extraction, the expected video location is:

```text
data/Something_data/videos/
```

Re-run the audit after extraction. The target is to make missing train/validation videos close to zero for the splits used in the paper.

### Diving48

Use the official UCSD SVCL page first. Download RGB videos and V2 train/test annotations when available. Recommended local layout:

```text
data/Diving48_data/
  raw_archives/
  videos/
  annotations/
```

After download/extraction, add a Diving48-specific manifest converter before training.

### FineGym

Use the official FineGym project page/GitHub for annotations and any released features. Treat FineGym as optional until the two priority datasets are stable.

## Acceptance Criteria Before Training

Do not launch large training until these are true:

1. Something-Something V2 has a manifest with missing/readable counts.
2. At least one temporal dataset has a class-balanced pilot subset.
3. The trusted evaluator has passed a smoke test on a small manifest.
4. Dataset paths and split definitions are frozen in `evaluations/accv2026/manifests/`.
5. The experiment tracker has been updated with dataset readiness.
