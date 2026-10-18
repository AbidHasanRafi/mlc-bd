"""Shared paths and constants for the MLC-BD pipeline.

Paths are relative to the repository, so the code runs from any clone.
Set MLCBD_DATA to the folder that contains the three provider folders
("ArcGIS Map", "Bing Maps", "Google Maps"). The default is <repo>/data/MLC-BD.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("MLCBD_DATA", REPO / "data" / "MLC-BD"))
OUT = REPO / "outputs"          # intermediate results (tables, small arrays; large feature files are git-ignored)
FIG = REPO / "figures"          # figure files written by the fig*.py scripts
META = REPO / "metadata"        # tables released with the images
GEO = REPO / "geo"              # Natural Earth boundary files (downloaded by scripts/download_geo.py)
for _p in (OUT, FIG, META):
    _p.mkdir(parents=True, exist_ok=True)

SOURCES = ["ArcGIS Map", "Bing Maps", "Google Maps"]
SRC_SHORT = {"ArcGIS Map": "ArcGIS", "Bing Maps": "Bing", "Google Maps": "Google"}
CLASSES = ["Agricultural_Land", "Beaches_Coastline", "Built_up_Areas",
           "Grasslands_Open_Land", "Hills_Mountains", "Trees_Forests", "Water_Bodies"]
CLS_LABEL = {"Agricultural_Land": "Agricultural land", "Beaches_Coastline": "Beaches/coastline",
             "Built_up_Areas": "Built-up areas", "Grasslands_Open_Land": "Grasslands/open land",
             "Hills_Mountains": "Hills/mountains", "Trees_Forests": "Trees/forests",
             "Water_Bodies": "Water bodies"}
SEED = 20260718
