"""Run the MLC-BD pipeline in dependency order.

    python scripts/run_all.py                    # everything (features take about an hour on 4 CPU cores)
    python scripts/run_all.py --skip-features    # reuse outputs/feat_*.npy
    python scripts/run_all.py --figures-only     # regenerate the five figures from outputs/
    python scripts/run_all.py --from 06_benchmark.py

Environment: set MLCBD_DATA to the folder with the three provider directories (default: data/MLC-BD).
Figure 1 needs the Natural Earth files: run scripts/download_geo.py once.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
ANALYSIS = ["01_inventory.py", "02_duplicates.py", "03_features_deep.py", "04_features_handcrafted.py",
            "05_extraction_order.py", "06_benchmark.py", "07_composition_stats.py", "08_label_screening.py",
            "09_spectra_color.py", "10_embedding_adaptation.py", "11_release_tables.py"]
FEATURES = {"03_features_deep.py", "04_features_handcrafted.py"}
# fig03 writes signature_stats.csv, which fig02 reads, so it runs first
FIGURES = ["fig03_provider_signatures.py", "fig04_benchmark.py", "fig05_label_screen.py", "fig01_study_area.py", "fig02_workflow.py"]

ap = argparse.ArgumentParser()
ap.add_argument("--skip-features", action="store_true", help="skip the two feature-extraction steps")
ap.add_argument("--figures-only", action="store_true", help="run only the figure scripts")
ap.add_argument("--from", dest="start", default=None, help="start at this script name")
args = ap.parse_args()

steps = FIGURES if args.figures_only else ANALYSIS + FIGURES
if args.skip_features:
    steps = [s for s in steps if s not in FEATURES]
if args.start:
    steps = steps[steps.index(args.start):]

for name in steps:
    t0 = time.time()
    print(f"\n=== {name}", flush=True)
    r = subprocess.run([sys.executable, str(SRC / name)])
    if r.returncode:
        sys.exit(f"{name} failed with exit code {r.returncode}")
    print(f"--- {name} finished in {time.time() - t0:.0f} s", flush=True)
