"""Download the Natural Earth vector layers used by src/fig01_study_area.py into geo/.

Natural Earth data are in the public domain (https://www.naturalearthdata.com). The files are fetched from the
nvkelso/natural-earth-vector mirror on GitHub.
"""
import sys
import urllib.request
from pathlib import Path

GEO = Path(__file__).resolve().parent.parent / "geo"
BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
LAYERS = [
    "ne_10m_admin_0_countries",
    "ne_10m_admin_1_states_provinces",
    "ne_10m_rivers_lake_centerlines",
    "ne_10m_geography_regions_polys",
    "ne_10m_lakes",
    "ne_10m_populated_places_simple",
]

GEO.mkdir(exist_ok=True)
for name in LAYERS:
    dest = GEO / f"{name}.geojson"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"have {dest.name}")
        continue
    print(f"downloading {name} ...", flush=True)
    try:
        urllib.request.urlretrieve(BASE + f"{name}.geojson", dest)
    except Exception as exc:
        print(f"failed: {exc}", file=sys.stderr)
        dest.unlink(missing_ok=True)
        sys.exit(1)
print("done")
