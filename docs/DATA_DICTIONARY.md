# Data dictionary

Tables in `metadata/` describe the 10,683 image files of MLC-BD. All tables are UTF-8 CSV with a header row. Paths use forward slashes and are relative to the dataset root (the folder that contains `ArcGIS Map/`, `Bing Maps/`, and `Google Maps/`).

## File naming

```
<Provider folder>/<Class folder>/<Provider>_<Class>_<index>.png
ArcGIS Map/Agricultural_Land/ArcGIS_Agricultural_Land_0001.png
```

`<index>` is a four-digit counter that starts at 0001 in each provider and class folder and follows the order in which the chips were originally extracted. Neighboring indices usually come from the same site (see "Splits" below).

| Provider folder | Layer | Short name used in tables and figures |
|---|---|---|
| `ArcGIS Map` | Esri World Imagery (ArcGIS Online) | ArcGIS |
| `Bing Maps` | Bing Maps Aerial | Bing |
| `Google Maps` | Google Maps Satellite | Google |

| Class folder | Class label in the article |
|---|---|
| `Agricultural_Land` | Agricultural land |
| `Beaches_Coastline` | Beaches/coastline |
| `Built_up_Areas` | Built-up areas |
| `Grasslands_Open_Land` | Grasslands/open land |
| `Hills_Mountains` | Hills/mountains |
| `Trees_Forests` | Trees/forests |
| `Water_Bodies` | Water bodies |

## `metadata.csv` (one row per image, 32 columns)

| Column | Type | Meaning |
|---|---|---|
| `file` | text | Path of the image relative to the dataset root. Unique key. |
| `provider` | text | Provider folder name. |
| `class` | text | Class folder name. This is the label assigned at extraction. |
| `width`, `height` | integer | Image size in pixels. Images are not resized. |
| `aspect_ratio` | float | `width / height`. |
| `png_mode` | text | PNG color mode. `RGBA` for every file. |
| `file_bytes` | integer | File size in bytes. |
| `sha256` | text | SHA-256 checksum of the file. |
| `mean_r`, `mean_g`, `mean_b` | float | Mean value of each channel, 0 to 255. |
| `brightness` | float | Mean of `mean_r`, `mean_g`, `mean_b`. |
| `laplacian_var_128px` | float | Sharpness: variance of the Laplacian of the gray image after resizing to 128 × 128 px (area interpolation). |
| `highfreq_energy_fraction` | float | Share of spectral energy beyond half the Nyquist frequency of the gray image (resized to 128 px if larger). |
| `entropy` | float | Shannon entropy of the gray-level histogram, in bits. |
| `spectral_slope` | float | Slope of the radial power spectrum on log–log axes, fitted between 0.05 and 0.44 cycles per pixel, from the central 64 × 64 px crop at native resolution (Hann window). Empty for the 11 chips whose shorter side is below 64 px or whose spectrum has zero power in the fit range. |
| `cielab_L`, `cielab_a`, `cielab_b` | float | CIELAB (D65) coordinates of the chip-mean RGB value. |
| `cielab_chroma` | float | `sqrt(a² + b²)`. |
| `duplicate_group_id` | integer | Connected component of the duplicate search (SHA-256 plus perceptual hashes, see `src/02_duplicates.py`). Chips that share an id are duplicates of each other. |
| `exact_duplicate_group_size` | integer | Size of that group. 1 for unique chips, 2 for the 16 chips in the 8 duplicate pairs. |
| `qc_defective_size` | boolean | Shorter side below 32 px. True for one file (`Google_Beaches_Coastline_0476.png`, 4 × 7 px). |
| `qc_small_side_lt100` | boolean | Shorter side below 100 px. |
| `split` | text | `train`, `val`, or `test`. See "Splits". |
| `oof_prob_given_label` | float | Out-of-fold probability that the averaged ResNet-50 and DINOv2-S linear probes assign to the label in `class`. |
| `oof_predicted_label` | text | Class with the highest out-of-fold probability. |
| `oof_suggested_label` | text | Class suggested by confident learning. Equals `class` when no other class clears its own threshold. |
| `oof_prob_suggested_label` | float | Out-of-fold probability of `oof_suggested_label`. |
| `label_issue_candidate` | boolean | Flagged by confident learning. True for 16 chips. |
| `oof_disagrees_with_label` | boolean | `oof_predicted_label` differs from `class`. True for 237 chips. |

The label-screening columns are suggestions for review. No label in the released folders was changed because of them.

## `splits.csv`

| Column | Meaning |
|---|---|
| `file`, `provider`, `class` | As in `metadata.csv`. |
| `split` | `train` (7,494 chips), `val` (1,600), or `test` (1,589). |

The split is about 70/15/15 and stratified by provider and class. The unit of assignment is a block of 25 consecutive extraction indices within one provider and class folder, merged with exact-duplicate groups. Blocks keep near neighbors together because consecutive chips are more similar than random pairs (cosine similarity of DINOv2 features 0.652 against 0.529). For a test that needs strict independence between training and test data, use leave-one-provider-out evaluation instead.

## `label_issue_candidates.csv`

The 16 chips flagged by confident learning.

| Column | Meaning |
|---|---|
| `file`, `provider`, `class` | As in `metadata.csv`. |
| `oof_suggested_label` | Suggested class. |
| `oof_prob_given_label` | Out-of-fold probability of the assigned class. |
| `oof_prob_suggested_label` | Out-of-fold probability of the suggested class. |

## Intermediate results in `outputs/`

The pipeline writes its intermediate tables to `outputs/`. The ones tracked in git are small result tables and arrays that the figure scripts read.

| File | Written by | Content |
|---|---|---|
| `inventory.csv` | `01_inventory` | Per-image audit: size, checksums, color and sharpness statistics, perceptual hashes. Row order defines the row order of all feature arrays. |
| `dup_group.csv`, `duplicate_pairs.csv` | `02_duplicates` | Duplicate groups and the pairs found. |
| `feat_resnet50.npy`, `feat_dinov2s.npy` | `03_features_deep` | Frozen-network features (not tracked; 87 MB and 16 MB). |
| `feat_handcrafted.npy` | `04_features_handcrafted` | 72-dimensional color–texture descriptor (not tracked). |
| `block_groups.csv`, `leakage_results.json` | `05_extraction_order` | Block groups for splitting and the extraction-order analysis. |
| `splits.csv`, `benchmark_results.csv`, `oof_*.npy`, `lopo_cm_*.npy`, `pooled_cm.npy`, `domain_*.csv` | `06_benchmark` | Splits, accuracy under four protocols, out-of-fold probabilities, confusion matrices, provider identification, Fréchet distances. |
| `composition_stats.json`, `table3_geometry.csv` | `07_composition_stats` | Balance tests, geometry summary (Table 6 of the article), duplicate-threshold sensitivity. |
| `label_issues.csv`, `label_issue_summary.json` | `08_label_screening` | Confident-learning output for every chip and a summary. |
| `spectral.csv`, `lab.csv`, `psd_rel.npy` | `09_spectra_color` | Spectral slope, CIELAB color, and radial spectra. |
| `silhouette.csv`, `centroids.csv`, `provider_cm.npy`, `adaptation_results.csv`, `adaptation_perclass.csv`, `learning_curves.csv` | `10_embedding_adaptation` | Embedding geometry, provider identification, z-score and CORAL baselines, learning curves. |
| `signature_stats.csv`, `signature_extra.json` | `fig03_provider_signatures` | Kruskal–Wallis statistics for the rendering signatures (read by `fig02_workflow`). |
| `centroid_structure.json` | `10_embedding_adaptation` | Centroid-distance summary quoted in the article. |
