"""Composition statistics, balance tests, duplicate-threshold sensitivity -> outputs/composition_stats.json, table3.csv"""
import sys, pathlib, json; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, pandas as pd
from scipy import stats
from common import *
inv = pd.read_csv(OUT / "inventory.csv"); res = {}
cls_tot = inv.groupby("cls").size().loc[CLASSES]; src_tot = inv.groupby("source").size().loc[SOURCES]
res["chi2_class_uniform"] = dict(zip(["chi2", "p"], map(float, stats.chisquare(cls_tot.values))))
res["chi2_source_uniform"] = dict(zip(["chi2", "p"], map(float, stats.chisquare(src_tot.values))))
ct = inv.groupby(["source", "cls"]).size().unstack().loc[SOURCES, CLASSES]
chi2, p, dof, _ = stats.chi2_contingency(ct.values); res["chi2_independence_source_class"] = dict(chi2=float(chi2), p=float(p), dof=int(dof), cramers_v=float(np.sqrt(chi2 / (ct.values.sum() * 2))))
res["class_total_min_max"] = [int(cls_tot.min()), int(cls_tot.max())]; res["imbalance_ratio_class"] = float(cls_tot.max() / cls_tot.min())
cell = inv.groupby(["source", "cls"]).size(); res["cell_min"] = [int(cell.min()), cell.idxmin()]; res["cell_max"] = [int(cell.max()), cell.idxmax()]
res["total_MiB"] = float(inv.bytes.sum() / 2**20); res["total_MB"] = float(inv.bytes.sum() / 1e6)
res["alpha_all_opaque"] = bool((inv.alpha_min == 255).all())
g = inv.assign(minside=inv[["width", "height"]].min(axis=1), ar=inv.width / inv.height)
rows = []
for s in SOURCES + ["All"]:
    d = g if s == "All" else g[g.source == s]
    rows.append(dict(provider=s, images=len(d), w_min=d.width.min(), w_med=d.width.median(), w_max=d.width.max(), h_min=d.height.min(), h_med=d.height.median(), h_max=d.height.max(),
                     mpix_median=d.pixels.median() / 1e3, below_100px=int((d.minside < 100).sum()), below_64px=int((d.minside < 64).sum()), below_32px=int((d.minside < 32).sum()),
                     ar_beyond_2=float(((d.ar > 2) | (d.ar < .5)).mean() * 100), size_kb_median=d.bytes.median() / 1024, sizeMiB=d.bytes.sum() / 2**20))
pd.DataFrame(rows).round(2).to_csv(OUT / "table3_geometry.csv", index=False); print(pd.DataFrame(rows).round(1).to_string())
# size by provider tests
w = [g[g.source == s].pixels.values for s in SOURCES]; res["kruskal_pixels"] = dict(zip(["H", "p"], map(float, stats.kruskal(*w))))
# duplicate threshold sensitivity (phash only)
def h2u(s): return np.array([int(x, 16) for x in s], dtype=np.uint64)
def pop(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555)); x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F); return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int16)
P = h2u(inv.phash); n = len(P); counts = {t: 0 for t in [0, 2, 4, 6, 8, 10]}; cross = {t: 0 for t in counts}
for a in range(0, n, 500):
    hp = pop(P[a:a + 500, None] ^ P[None, :]); r, c = np.nonzero(np.triu(np.ones_like(hp, bool), k=1 - a)[: hp.shape[0]] if False else (hp <= 10))
    keep = (r + a) < c; r, c = r[keep] + a, c[keep]; d = hp[r - a, c]
    for t in counts:
        m = d <= t; counts[t] += int(m.sum()); cross[t] += int((m & ((inv.cls.values[r] != inv.cls.values[c]) | (inv.source.values[r] != inv.source.values[c]))).sum())
res["phash_pairs_by_threshold"] = counts; res["phash_pairs_cross_class_or_source"] = cross
json.dump(res, open(OUT / "composition_stats.json", "w"), indent=1); print(json.dumps(res, indent=1))
