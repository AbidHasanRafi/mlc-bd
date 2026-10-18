"""Fig. 1 - (a) study area with divisions, rivers and city population; (b) land-cover context of the seven classes with example chips.
Natural Earth 1:10m (public domain). Per-chip coordinates were not recorded, so no chip locations are plotted; regions in (b) are
representative settings, with dashed outlines marking generalised (not surveyed) zones."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from shapely.geometry import Polygon, box
from PIL import Image
from style import *

EXT = (87.7, 20.4, 93.9, 27.0)
clip = box(*EXT)
adm0 = gpd.read_file(GEO / "ne_10m_admin_0_countries.geojson")
adm1 = gpd.read_file(GEO / "ne_10m_admin_1_states_provinces.geojson")
rivers = gpd.read_file(GEO / "ne_10m_rivers_lake_centerlines.geojson")
regions = gpd.read_file(GEO / "ne_10m_geography_regions_polys.geojson")
lakes = gpd.read_file(GEO / "ne_10m_lakes.geojson")
places = gpd.read_file(GEO / "ne_10m_populated_places_simple.geojson")
bgd = adm0[adm0["ADMIN"] == "Bangladesh"]
nb = adm0[adm0["ADMIN"].isin(["India", "Myanmar", "Nepal", "Bhutan"])]
div = adm1[adm1["admin"] == "Bangladesh"].copy()
city = places[(places.adm0name == "Bangladesh") & (places.pop_max > 200000)].copy()

WATER_BG, LAND, BEDGE = "#e9f1fa", "#e8e7e2", "#2b2b2b"
RIVER = "#2f6fa8"
TINT = ["#f6e3d0", "#e4efd6", "#dfe6f4", "#f4dfe6", "#e9e2f2", "#f3ecc8", "#d9eeee"]
CC = dict(water="#2f6fa8", forest="#2f6b3a", agri="#c9a227", grass="#93a832", hills="#8a5a2b", beach="#e0a458", built="#b1473d")
RENAME = {"Chittagong": "Chattogram", "Barisal": "Barishal"}


def river(name=None, idx=None):
    g = rivers[rivers["name"] == name] if name else rivers.loc[[idx]]
    return gpd.clip(g, clip)


ganges, brahma, tista = river("Ganges"), river("Brahmaputra"), river("Tista")
meghna, karna = river(idx=840), river(idx=169)
sundarbans = regions[regions["NAME"] == "Sundarbans"]
chin = gpd.clip(regions[regions["NAME"] == "Chin Hills"], clip)
kaptai, haor = lakes.loc[[1227]], lakes.loc[[678]]
char_band = gpd.clip(brahma, bgd).buffer(0.09)
agri_belt = Polygon([(88.05, 24.35), (88.35, 25.85), (88.9, 26.35), (89.55, 26.2), (89.75, 25.55), (89.35, 24.75), (88.75, 24.15), (88.05, 24.35)])


def base(ax, labels=True):
    ax.set_xlim(EXT[0], EXT[2]); ax.set_ylim(EXT[1], EXT[3]); ax.set_facecolor(WATER_BG)
    ax.set_aspect(1 / np.cos(np.radians(23.5)))
    nb.plot(ax=ax, facecolor=LAND, edgecolor="#a9a79e", lw=0.6, zorder=1)
    ax.set_xticks(range(88, 94)); ax.set_yticks(range(21, 27))
    ax.set_xticklabels([f"{v}°E" for v in range(88, 94)], fontsize=FS_TICK - 1)
    ax.set_yticklabels([f"{v}°N" for v in range(21, 27)] if labels else [], fontsize=FS_TICK - 1)
    frame(ax, minor=False)
    for t, (x, y) in {"INDIA": (90.7, 26.45), "MYANMAR": (93.15, 21.55)}.items():
        ax.text(x, y, t, fontsize=FS_ANN - 0.5, style="italic", color="#6b6a63", ha="center", zorder=2)


def scalebar(ax, x0, y0, km=100):
    L = km / (111.32 * np.cos(np.radians(23.5)))
    ax.plot([x0, x0 + L], [y0, y0], color=INK, lw=2.4, solid_capstyle="butt", zorder=9)
    ax.plot([x0, x0 + L / 2], [y0, y0], color="white", lw=1.2, zorder=10)
    ax.text(x0 + L / 2, y0 + 0.1, f"{km} km", fontsize=FS_ANN, ha="center", zorder=9)


def north(ax, x, y):
    ax.annotate("N", xy=(x, y + 0.55), xytext=(x, y), fontsize=FS_LEG, fontweight="bold", ha="center", arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2), zorder=9)


halo = [pe.withStroke(linewidth=2.2, foreground="white")]
fig = plt.figure(figsize=(W2, 4.55))
H = 3.58 / 4.55
axa = fig.add_axes([0.075, 0.085, 0.37, H]); axb = fig.add_axes([0.49, 0.085, 0.37, H])

# ---- (a) study area
base(axa)
for k, (_, r) in enumerate(div.iterrows()):
    gpd.GeoSeries([r.geometry], crs=div.crs).plot(ax=axa, facecolor=TINT[k % 7], edgecolor="#7a6a55", lw=0.7, zorder=2)
bgd.boundary.plot(ax=axa, edgecolor=BEDGE, lw=1.5, zorder=3)
for g_, lw in [(ganges, 1.5), (brahma, 1.5), (meghna, 1.3), (karna, 1.0), (tista, 0.9)]:
    g_.plot(ax=axa, color=RIVER, lw=lw, zorder=4)
for _, r in div.iterrows():
    p = r.geometry.representative_point()
    axa.text(p.x, p.y + 0.22, RENAME.get(r["name"], r["name"]), fontsize=FS_ANN, fontweight="bold", ha="center", va="center", color="#3c3226", zorder=6, path_effects=halo)
for _, r in city.iterrows():
    axa.scatter(r.geometry.x, r.geometry.y, s=14 + 105 * np.sqrt(r.pop_max / 1.3e7), fc="#c0392b", ec="white", lw=0.7, zorder=7, alpha=0.95)
sizes = [(1e6, "1 M"), (5e6, "5 M"), (1.3e7, "13 M")]
h = [Line2D([], [], marker="o", ls="", mfc="#c0392b", mec="white", ms=np.sqrt(14 + 105 * np.sqrt(v / 1.3e7)), label=t) for v, t in sizes]
leg(axa, "lower center", handles=h, fontsize=FS_ANN, title="City population", title_fontsize=FS_ANN, ncol=3, handlelength=0.8,
    borderpad=0.35, labelspacing=0.2, columnspacing=1.0, handletextpad=0.3)
scalebar(axa, 92.75, 24.35); north(axa, 93.45, 23.2)
axa.set_ylabel("Latitude"); axa.set_xlabel("Longitude"); axb.set_xlabel("Longitude")
panel(axa, "a", dx=-0.17, dy=1.01)
ins = fig.add_axes([0.318, 0.70, 0.122, 0.168])
sa = adm0[adm0["ADMIN"].isin(["Afghanistan", "Pakistan", "India", "Nepal", "Bhutan", "Bangladesh", "Sri Lanka", "Myanmar"])]
sa.plot(ax=ins, facecolor=LAND, edgecolor="#a9a79e", lw=0.3); bgd.plot(ax=ins, facecolor="#c0392b", edgecolor="#c0392b", lw=0.3)
ins.set_xticks([]); ins.set_yticks([]); ins.set_facecolor(WATER_BG); ins.set_aspect(1 / np.cos(np.radians(22)))
for sp in ins.spines.values():
    sp.set_visible(True); sp.set_linewidth(0.9)
ins.tick_params(which="both", length=0)

# ---- (b) land-cover context
base(axb, labels=False)
bgd.plot(ax=axb, facecolor="#f4f3ee", edgecolor=BEDGE, lw=1.3, zorder=2)
gpd.GeoSeries([agri_belt], crs=bgd.crs).plot(ax=axb, facecolor=CC["agri"], alpha=0.38, edgecolor=CC["agri"], lw=1.2, linestyle="--", zorder=2.4)
char_band.plot(ax=axb, facecolor=CC["grass"], alpha=0.42, edgecolor=CC["grass"], lw=1.0, linestyle="--", zorder=2.5)
sundarbans.plot(ax=axb, facecolor=CC["forest"], alpha=0.55, edgecolor=CC["forest"], lw=1.1, hatch="....", zorder=2.6)
chin.plot(ax=axb, facecolor=CC["hills"], alpha=0.42, edgecolor=CC["hills"], lw=1.1, hatch="//", zorder=2.6)
kaptai.plot(ax=axb, facecolor=CC["water"], edgecolor=CC["water"], lw=0.8, zorder=2.7)
haor.plot(ax=axb, facecolor=CC["water"], alpha=0.6, edgecolor=CC["water"], lw=0.8, zorder=2.7)
for g_, lw in [(ganges, 2.2), (brahma, 2.2), (meghna, 1.8), (karna, 1.4), (tista, 1.2)]:
    g_.plot(ax=axb, color=CC["water"], lw=lw, zorder=3)
bd = bgd.boundary.iloc[0]
pts = [(x, y) for x, y in bd.coords] if bd.geom_type == "LineString" else [pt for part in bd.geoms for pt in part.coords]
seg = sorted(set([(x, y) for x, y in pts if 91.85 <= x <= 92.35 and 20.70 <= y <= 21.55]), key=lambda p: -p[1])
axb.plot(*zip(*seg), color=CC["beach"], lw=3.6, solid_capstyle="round", zorder=3.2)
for _, r in city.iterrows():
    axb.scatter(r.geometry.x, r.geometry.y, s=14 + 105 * np.sqrt(r.pop_max / 1.3e7), fc=CC["built"], ec="white", lw=0.7, zorder=5)


def lab(t, xy, xyt, col, ha="left"):
    axb.annotate(t, xy=xy, xytext=xyt, fontsize=FS_ANN, fontweight="bold", ha=ha, va="center", zorder=8, path_effects=halo,
                 arrowprops=dict(arrowstyle="-", color=col, lw=0.9, shrinkA=1, shrinkB=1))


lab("Sundarbans", (89.2, 22.1), (88.05, 21.15), CC["forest"])
lab("Chittagong\nHill Tracts", (92.4, 23.2), (93.75, 24.35), CC["hills"], ha="right")
lab("Kaptai Lake", (92.25, 22.75), (93.7, 21.95), CC["water"], ha="right")
lab("Haor basin", (91.1, 24.15), (92.45, 25.6), CC["water"])
lab("Cox's Bazar–Teknaf", (92.1, 21.1), (92.05, 20.62), CC["beach"], ha="right")
lab("Agricultural\nplains", (89.05, 25.3), (88.0, 26.7), CC["agri"])
lab("Jamuna char\nlands", (89.75, 24.9), (91.15, 26.1), CC["grass"])
axb.text(88.55, 23.6, "Ganges–Padma", fontsize=FS_ANN - 0.5, color=CC["water"], style="italic", fontweight="bold", rotation=-28, ha="center", path_effects=halo, zorder=8)
scalebar(axb, 90.4, 21.1)
panel(axb, "b", dx=-0.05, dy=1.01)

# ---- chip legend column (one random chip per class and provider, centre-cropped squares)
ok = pd.read_csv(OUT / "inventory.csv")
ok = ok[ok[["width", "height"]].min(axis=1) >= 150]
rng = np.random.default_rng(SEED + 1)
cls_sw = [("Agricultural_Land", CC["agri"], "Agricultural\nland"), ("Beaches_Coastline", CC["beach"], "Beaches/\ncoastline"), ("Built_up_Areas", CC["built"], "Built-up\nareas"),
          ("Grasslands_Open_Land", CC["grass"], "Grasslands/\nopen land"), ("Hills_Mountains", CC["hills"], "Hills/\nmountains"), ("Trees_Forests", CC["forest"], "Trees/\nforests"),
          ("Water_Bodies", CC["water"], "Water\nbodies")]
x0, top, rowh = 0.872, 0.955, 0.138
th = 0.0535
tw = th * 4.55 / 7.16
for j, s_ in enumerate(SOURCES):
    fig.text(x0 + 0.013 + j * tw * 1.04 + tw / 2, top + 0.012, SRC_SHORT[s_][0], fontsize=FS_ANN, color=PROV[s_], fontweight="bold", ha="center", va="bottom")
for i, (c, col, nm) in enumerate(cls_sw):
    yt = top - i * rowh
    fig.add_artist(mpatches.Rectangle((x0, yt - 0.118), 0.007, 0.118, transform=fig.transFigure, fc=col, ec="none"))
    fig.text(x0 + 0.013, yt - 0.003, nm, fontsize=FS_ANN - 0.5, va="top", ha="left", fontweight="bold", linespacing=1.05)
    for j, s_ in enumerate(SOURCES):
        r = ok[(ok.cls == c) & (ok.source == s_)].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        im = Image.open(DATA / r.relpath).convert("RGB"); w, h_ = im.size; m = min(w, h_)
        im = im.crop(((w - m) // 2, (h_ - m) // 2, (w - m) // 2 + m, (h_ - m) // 2 + m)).resize((160, 160), Image.LANCZOS)
        a_ = fig.add_axes([x0 + 0.013 + j * (tw * 1.04), yt - 0.062 - th, tw, th])
        a_.imshow(im); a_.set_xticks([]); a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_visible(True); sp.set_linewidth(1.5); sp.set_edgecolor(PROV[s_])
        a_.tick_params(which="both", length=0)
save(fig, "Fig1_study_area")
