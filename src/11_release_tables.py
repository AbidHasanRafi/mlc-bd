"""Assemble the tables released next to the images (not the images themselves), written to metadata/:
  metadata.csv       one row per image: identity, geometry, checksums, QC flags, split, label-screening result
  splits.csv         file -> train/val/test (also embedded in metadata.csv)
  label_issues.csv   screening output for flagged images only
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, pandas as pd
from common import *

REL = META
inv = pd.read_csv(OUT / "inventory.csv")
sp = pd.read_csv(OUT / "splits.csv")[["relpath", "split", "fold20", "dup_group"]]
li = pd.read_csv(OUT / "label_issues.csv")[["relpath", "suggested", "p_given", "p_suggested", "flag_cl", "argmax"]]
spc = pd.read_csv(OUT / "spectral.csv")[["relpath", "spec_slope"]]
lab_ = pd.read_csv(OUT / "lab.csv").rename(columns={"L": "cielab_L", "a": "cielab_a", "b": "cielab_b"})[["relpath", "cielab_L", "cielab_a", "cielab_b", "chroma"]]
m = inv.merge(sp, on="relpath").merge(li, on="relpath").merge(spc, on="relpath").merge(lab_, on="relpath")
sizes = m.groupby("dup_group").relpath.transform("size")
m["exact_duplicate_group_size"] = sizes
m["qc_defective_size"] = (m[["width", "height"]].min(axis=1) < 32)
m["qc_small_side_lt100"] = (m[["width", "height"]].min(axis=1) < 100)
m["aspect_ratio"] = (m.width / m.height).round(3)
cols = ["relpath", "source", "cls", "width", "height", "aspect_ratio", "mode", "bytes", "sha256", "mean_r", "mean_g", "mean_b", "brightness",
        "lap_var_128", "hf_energy_ratio", "entropy", "spec_slope", "cielab_L", "cielab_a", "cielab_b", "chroma", "dup_group", "exact_duplicate_group_size", "qc_defective_size", "qc_small_side_lt100",
        "split", "p_given", "argmax", "suggested", "p_suggested", "flag_cl"]
out = m[cols].rename(columns={"relpath": "file", "source": "provider", "cls": "class", "mode": "png_mode", "bytes": "file_bytes", "dup_group": "duplicate_group_id",
                              "lap_var_128": "laplacian_var_128px", "hf_energy_ratio": "highfreq_energy_fraction", "spec_slope": "spectral_slope", "chroma": "cielab_chroma", "p_given": "oof_prob_given_label",
                              "argmax": "oof_predicted_label", "suggested": "oof_suggested_label", "p_suggested": "oof_prob_suggested_label", "flag_cl": "label_issue_candidate"})
out["oof_disagrees_with_label"] = out["oof_predicted_label"] != out["class"]
for c in ["mean_r", "mean_g", "mean_b", "brightness", "laplacian_var_128px", "highfreq_energy_fraction", "entropy", "spectral_slope", "cielab_L", "cielab_a", "cielab_b", "cielab_chroma", "oof_prob_given_label", "oof_prob_suggested_label"]:
    out[c] = out[c].round(4)
out.to_csv(REL / "metadata.csv", index=False)
out[["file", "provider", "class", "split"]].to_csv(REL / "splits.csv", index=False)
out[out.label_issue_candidate][["file", "provider", "class", "oof_suggested_label", "oof_prob_given_label", "oof_prob_suggested_label"]].to_csv(REL / "label_issue_candidates.csv", index=False)
print(out.shape, out.split.value_counts().to_dict(), "candidates", int(out.label_issue_candidate.sum()), "disagree", int(out.oof_disagrees_with_label.sum()))
