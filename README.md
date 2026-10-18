# MLC-BD: analysis code

MLC-BD contains 10,683 RGB image chips of Bangladesh in seven land-cover classes. Every class was sampled separately from three web-mapping platforms (Esri World Imagery, Bing Maps Aerial, and Google Maps Satellite), so imagery provider is crossed with class. This repository holds everything needed to regenerate the numbers, tables, and figures of the article from the images. **It does not contain the images.**

## Contents

1. [What is here](#1-what-is-here)
2. [Repository layout](#2-repository-layout)
3. [Quick start](#3-quick-start)
4. [Pipeline](#4-pipeline)
5. [Figures](#5-figures)
6. [Released tables](#6-released-tables)
7. [Results you should reproduce](#7-results-you-should-reproduce)
8. [Data and licensing](#8-data-and-licensing)

## 1. What is here

| Item | In this repository |
|---|---|
| Image files (10,683 PNG, 755 MiB) | No. Download from the data repository (section 8) and place them under `data/MLC-BD/`. |
| Per-image metadata, fixed splits, label-screening result | Yes, in `metadata/` (described in [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md)). |
| Analysis code (inventory, duplicates, features, benchmark, label screening, figures) | Yes, in `src/`. |
| Intermediate result tables and small arrays | Yes, in `outputs/`, so the figures can be rebuilt without the 45-minute feature step. |
| Figure sources | Yes, in `figures/` (PDF, plus the editable `Fig2_workflow.drawio`). PNG renders are git-ignored and rebuilt by the scripts. |
| Manuscript text | No. |

The images are renderings of imagery owned by the platform operators and their suppliers. Check the terms in section 8 before redistributing them.

## 2. Repository layout

```
.
├── README.md
├── LICENSE                     MIT license for the code (not for the images)
├── requirements.txt            pinned versions used for the article
├── metadata/
│   ├── metadata.csv            one row per image, 32 columns
│   ├── splits.csv              train / val / test assignment
│   └── label_issue_candidates.csv
├── src/
│   ├── common.py               paths and constants (reads MLCBD_DATA)
│   ├── style.py                figure style shared by the fig*.py scripts
│   ├── 01_inventory.py ... 11_release_tables.py     analysis pipeline
│   └── fig01_study_area.py ... fig05_label_screen.py  the five figures
├── scripts/
│   ├── run_all.py              runs the pipeline in dependency order
│   └── download_geo.py         fetches the Natural Earth layers for Fig. 1
├── outputs/                    intermediate results (see docs/DATA_DICTIONARY.md)
└── docs/DATA_DICTIONARY.md     every column and file explained
```

## 3. Quick start

Requirements: Python 3.14.3 was used for the article; other recent versions should work but have not been tested. Plan for a few GB of free disk space beyond the images and a CPU with at least 4 cores. No GPU is needed. Internet access is needed once, to download model weights and the boundary files.

```bash
# 1. environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. data: download MLC-BD (section 8) and unpack so that this path exists:
#    data/MLC-BD/ArcGIS Map/Agricultural_Land/ArcGIS_Agricultural_Land_0001.png
#    or keep the images elsewhere and point to them:
export MLCBD_DATA="/path/to/MLC-BD"  # Windows PowerShell: $env:MLCBD_DATA = "D:\MLC-BD"

# 3. boundary files for Fig. 1 (once)
python scripts/download_geo.py

# 4. run everything
python scripts/run_all.py
```

Useful variants:

```bash
python scripts/run_all.py --skip-features        # reuse outputs/feat_*.npy (after one full run)
python scripts/run_all.py --figures-only         # redraw the figures from the tracked outputs/
python scripts/run_all.py --from 06_benchmark.py # resume from a step
python src/06_benchmark.py                       # any step also runs on its own
```

`--figures-only` needs the images only for the example chips in Fig. 1 and Fig. 5, and the boundary files for Fig. 1. Figs. 3 and 4 and the workflow diagram need nothing but `outputs/`.

Measured on the machine used for the article, the whole pipeline takes about an hour, almost all of it in step 3. Without that step it takes about 11 minutes.

The first run downloads the ResNet-50 weights (`IMAGENET1K_V2`, 98 MB) from the PyTorch servers and the DINOv2-small weights (`facebook/dinov2-small`) from the Hugging Face Hub.

## 4. Pipeline

Steps must run in the order shown, because later steps read the files that earlier steps write. `scripts/run_all.py` does this. Times are for a 4-core CPU without a GPU.

| Step | Script | Reads | Writes (in `outputs/`) | Time |
|---|---|---|---|---|
| 1 | `01_inventory.py` | images | `inventory.csv`: decode check, size, alpha channel, SHA-256, perceptual hashes, color and sharpness statistics | about 1.5 min |
| 2 | `02_duplicates.py` | `inventory.csv` | `dup_group.csv`, `duplicate_pairs.csv`: exact duplicates (SHA-256) and near duplicates (perceptual hashes) | 15 s |
| 3 | `03_features_deep.py` | images | `feat_resnet50.npy` (2,048-d), `feat_dinov2s.npy` (384-d) | about 45 min |
| 4 | `04_features_handcrafted.py` | images | `feat_handcrafted.npy` (72-d color–texture descriptor) | 2 min |
| 5 | `05_extraction_order.py` | features, `dup_group.csv` | `block_groups.csv`, `leakage_results.json`: tests whether extraction order leaks information across splits | 2.5 min |
| 6 | `06_benchmark.py` | features, `block_groups.csv` | `splits.csv`, `benchmark_results.csv`, `oof_*.npy`, confusion matrices, `domain_*.csv`: pooled, within-provider, one-to-another, and leave-one-provider-out probes; out-of-fold probabilities; provider identification; Fréchet distances | 1.5 min |
| 7 | `07_composition_stats.py` | `inventory.csv` | `composition_stats.json`, `table3_geometry.csv`: balance tests and geometry summary | 10 s |
| 8 | `08_label_screening.py` | `oof_ensemble.npy` | `label_issues.csv`, `label_issue_summary.json`: confident-learning screen | 2 s |
| 9 | `09_spectra_color.py` | images, `inventory.csv` | `spectral.csv`, `lab.csv`, `psd_rel.npy`: radial power spectra, spectral slope, CIELAB color | 1.5 min |
| 10 | `10_embedding_adaptation.py` | features, `splits.csv`, `block_groups.csv` | `silhouette.csv`, `centroids.csv`, `centroid_structure.json`, `provider_cm.npy`, `adaptation_*.csv`, `learning_curves.csv`: embedding geometry, z-score and CORAL baselines, learning curves | 1 min |
| 11 | `11_release_tables.py` | all of the above | `metadata/metadata.csv`, `metadata/splits.csv`, `metadata/label_issue_candidates.csv` | 2 s |

Method details (thresholds, window sizes, hyperparameters) are in the docstring at the top of each script and in the Technical Validation section of the article. The main design choices:

- **Splits.** Files are indexed in extraction order, and consecutive chips are more similar than random pairs. Splits therefore assign blocks of 25 consecutive indices (merged with duplicate groups) to one side, stratified by provider and class (`05_extraction_order.py`, `06_benchmark.py`).
- **Classifiers.** Multinomial logistic regression on standardized frozen features, with the regularization strength chosen on the validation split from {0.01, 0.1, 1}.
- **Seed.** `SEED = 20260718` in `src/common.py` drives every random choice.

## 5. Figures

| Figure | Script | Needs | Output |
|---|---|---|---|
| Fig. 1 Study area, land-cover context, example chips | `fig01_study_area.py` | `geo/` (run `download_geo.py`), `outputs/inventory.csv`, images | `figures/Fig1_study_area.{png,pdf}` |
| Fig. 2 Workflow | `fig02_workflow.py` | `outputs/` (run `fig03` first) | `figures/Fig2_workflow.{png,pdf,drawio}` |
| Fig. 3 Provider rendering signatures | `fig03_provider_signatures.py` | `outputs/` | `figures/Fig3_provider_signatures.{png,pdf}`, `outputs/signature_*.{csv,json}` |
| Fig. 4 Cross-provider benchmark | `fig04_benchmark.py` | `outputs/` | `figures/Fig4_benchmark.{png,pdf}` |
| Fig. 5 Label screening | `fig05_label_screen.py` | `outputs/`, images | `figures/Fig5_label_screen.{png,pdf}` |

All figures are 600 dpi PNG plus vector PDF at 7.16 in (IEEE/ACM double-column) width. `fig02_workflow.py` also writes `Fig2_workflow.drawio`, which opens in [diagrams.net](https://app.diagrams.net). Re-running the script overwrites edits made in that file.

## 6. Released tables

`metadata/metadata.csv` has one row per image: provider, class, size, checksum, color and sharpness statistics, duplicate group, quality flags, split, and the label-screening result. `metadata/splits.csv` is the same split in a short form, and `metadata/label_issue_candidates.csv` lists the 16 chips flagged for review. Every column is defined in [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

Dataset counts (images per class and provider):

| Class | ArcGIS | Bing | Google | Total |
|---|---:|---:|---:|---:|
| Agricultural land | 506 | 525 | 516 | 1,547 |
| Beaches/coastline | 515 | 517 | 506 | 1,538 |
| Built-up areas | 505 | 515 | 500 | 1,520 |
| Grasslands/open land | 504 | 505 | 500 | 1,509 |
| Hills/mountains | 519 | 545 | 456 | 1,520 |
| Trees/forests | 503 | 506 | 517 | 1,526 |
| Water bodies | 504 | 519 | 500 | 1,523 |
| **Total** | **3,556** | **3,632** | **3,495** | **10,683** |

## 7. Results you should reproduce

A full run should give these values. Differences in the last digit can occur with other library versions or CPUs.

| Quantity | Value | Source file |
|---|---|---|
| Files that decode | 10,683 of 10,683 | `inventory.csv` |
| Exact duplicate pairs | 8 (16 chips) | `duplicate_pairs.csv` |
| Defective file | 1 (`Google_Beaches_Coastline_0476.png`, 4 × 7 px) | `inventory.csv` |
| Median sharpness (Laplacian variance), ArcGIS / Bing / Google | 73 / 34 / 143 | `signature_stats.csv` |
| Median CIELAB b\*, ArcGIS / Bing / Google | 15.2 / 14.5 / −2.6 | `signature_stats.csv` |
| Provider identified from a chip (DINOv2-S, out-of-fold) | 90.8 % | `provider_cm.npy` |
| DINOv2-S probe, pooled test split | 95.6 % | `benchmark_results.csv` |
| DINOv2-S probe, within provider (mean) | 96.7 % | `benchmark_results.csv` |
| DINOv2-S probe, one provider to another (mean of six pairs) | 79.9 % | `benchmark_results.csv` |
| Leave one provider out, ArcGIS / Bing / Google unseen | 80.3 / 90.4 / 73.9 % | `benchmark_results.csv` |
| Same, with per-provider z-scoring (mean of three) | 82.7 % (from 81.4 %) | `adaptation_results.csv` |
| Out-of-fold disagreements with the assigned label | 237 chips (2.2 %) | `label_issues.csv` |
| Confident-learning candidates | 16 chips | `label_issue_summary.json` |

## 8. Data and licensing

**Images.** The dataset images are hosted on Mendeley Data (https://doi.org/10.17632/gr549zm73h.1).

**Terms of the imagery.** Each chip is a rendering of imagery that Esri, Microsoft, Google, and their imagery suppliers own. The platforms set their own terms for copying and redistribution.

**Code, metadata tables, and figures.** The code is released under the MIT license (see `LICENSE`).

**Boundary data.** Natural Earth layers are in the public domain.