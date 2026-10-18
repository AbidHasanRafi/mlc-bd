"""Fig. 3 - provider rendering signatures: spectra, sharpness, CIELAB colour, colour difference.
Also writes outputs/signature_stats.csv and outputs/signature_extra.json (numbers quoted in the text)."""
import sys, pathlib, json; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, pandas as pd, seaborn as sns
from scipy import stats
from matplotlib.patches import Ellipse
from style import *

inv = pd.read_csv(OUT / "inventory.csv")
spc = pd.read_csv(OUT / "spectral.csv")
lab = pd.read_csv(OUT / "lab.csv")
P = np.load(OUT / "psd_rel.npy")
d = inv.merge(spc, on="relpath").merge(lab, on="relpath")
d["log_lap"] = np.log10(d.lap_var_128 + 1e-6)
ok = (d[["width", "height"]].min(axis=1) >= 32).values          # excludes the one defective file
tot = d.mean_r + d.mean_g + d.mean_b
d["chr_b"] = d.mean_b / tot

rows = []
for m in ["lap_var_128", "hf_energy_ratio", "brightness", "chr_b", "spec_slope", "L", "a", "b", "chroma"]:
    sub = d[ok & d[m].notna()]
    g = [sub.loc[sub.source == s, m].values for s in SOURCES]
    H, p = stats.kruskal(*g); n = sum(map(len, g))
    rows.append(dict(metric=m, H=H, p=p, epsilon_sq=(H - 2) / (n - 3), **{f"median_{SRC_SHORT[s]}": np.median(x) for s, x in zip(SOURCES, g)}))
pd.DataFrame(rows).to_csv(OUT / "signature_stats.csv", index=False)
ST = pd.DataFrame(rows).set_index("metric")

fig, axs = plt.subplots(2, 2, figsize=(W2, 6.2), gridspec_kw=dict(hspace=0.36, wspace=0.42, left=0.095, right=0.985, top=0.965, bottom=0.085))
f = np.arange(1, 32) / 64.0

# (a) radial power spectrum
ax = axs[0, 0]; panel(ax, "a", dx=-0.17)
for s in SOURCES:
    m = (inv.source == s).values & ok & ~np.isnan(P[:, 0])
    q = np.nanpercentile(P[m], [25, 50, 75], axis=0)
    ax.fill_between(f, q[0], q[2], color=PROV[s], alpha=0.18, lw=0)
    ax.plot(f, q[1], color=PROV[s], lw=2.2, label=f"{SRC_SHORT[s]}  ($\\beta$ = {ST.loc['spec_slope', 'median_' + SRC_SHORT[s]]:.2f})")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1 / 64, 0.5)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5]); ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"])
ax.set_xlabel("Spatial frequency (cycles/px)"); ax.set_ylabel("Relative power spectral density"); frame(ax, minor=False)
leg(ax, "lower left", fontsize=FS_ANN, handlelength=1.4, handletextpad=0.3, borderpad=0.3)

# (b) sharpness
ax = axs[0, 1]; panel(ax, "b", dx=-0.17)
sub = d[ok]
for k, s in enumerate(SOURCES):
    x = sub.loc[sub.source == s, "log_lap"].values
    sns.violinplot(y=x, x=np.full(len(x), k), ax=ax, color=PROV[s], inner=None, cut=0, linewidth=0.9, width=0.85, density_norm="width", saturation=1)
for coll in ax.collections:
    coll.set_alpha(0.45)
for k, s in enumerate(SOURCES):
    x = sub.loc[sub.source == s, "log_lap"].values; q1, q2, q3 = np.percentile(x, [25, 50, 75])
    ax.plot([k, k], [q1, q3], color=INK, lw=2.4, zorder=5); ax.plot(k, q2, "o", color="white", mec=INK, ms=5.5, mew=1.0, zorder=6)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels([SRC_SHORT[s] for s in SOURCES]); ax.set_xlim(-0.6, 2.6); ax.set_ylim(-0.8, 4.4)
ax.set_ylabel("Laplacian variance (log$_{10}$)"); frame(ax, minor=False); ax.tick_params(axis="x", which="both", bottom=False, top=False)
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
r = ST.loc["lap_var_128"]
note(ax, f"H = {r.H:,.0f}\n$\\varepsilon^2$ = {r.epsilon_sq:.2f}", "upper left", fs=FS_ANN)

# (c) CIELAB a*-b*
ax = axs[1, 0]; panel(ax, "c", dx=-0.17)
for s in SOURCES:
    m = (d.source == s).values & ok
    ax.scatter(d.a[m], d.b[m], s=2.5, color=PROV[s], alpha=0.12, lw=0, rasterized=True)
for s in SOURCES:
    m = (d.source == s).values & ok
    X = np.c_[d.a[m], d.b[m]]; mu = np.median(X, axis=0); C = np.cov(X.T)
    w, V = np.linalg.eigh(C); ang = np.degrees(np.arctan2(V[1, 1], V[0, 1])); k = np.sqrt(stats.chi2.ppf(0.5, 2))
    ax.add_patch(Ellipse(mu, 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(w[0]), angle=ang, fc="none", ec=PROV[s], lw=2.0))
    mk(ax, mu[0], mu[1], s, ms=8, label=SRC_SHORT[s], zorder=5)
ax.axhline(0, color="#999999", lw=0.7, zorder=0); ax.axvline(0, color="#999999", lw=0.7, zorder=0)
ax.set_xlim(-24, 24); ax.set_ylim(-14, 38)
ax.set_xlabel("CIELAB $a^*$ (green to red)"); ax.set_ylabel("CIELAB $b^*$ (blue to yellow)"); frame(ax)
leg(ax, "upper left", fontsize=FS_ANN, handletextpad=0.1, borderpad=0.3)

# (d) Delta E between provider medians within class
ax = axs[1, 1]; panel(ax, "d", dx=-0.42)
med = d[ok].groupby(["cls", "source"])[["L", "a", "b"]].median()
pairs = [("ArcGIS Map", "Bing Maps"), ("ArcGIS Map", "Google Maps"), ("Bing Maps", "Google Maps")]
dE = np.array([[np.linalg.norm(med.loc[(c, p1)].values - med.loc[(c, p2)].values) for p1, p2 in pairs] for c in CLASSES])
im = heat(ax, dE, vmin=0, vmax=40, fmt="{:.1f}", thresh=22, fs=FS_ANN + 0.5)
ax.set_xticklabels(["ArcGIS–\nBing", "ArcGIS–\nGoogle", "Bing–\nGoogle"], fontsize=FS_TICK - 1)
ax.set_yticklabels(SHORT2)
ax.set_title("$\\Delta E^*_{ab}$ between provider medians", fontsize=FS_TICK, pad=4)
cbar(fig, ax, im, "$\\Delta E^*_{ab}$")
save(fig, "Fig3_provider_signatures")
json.dump(dict(dE=dE.round(1).tolist(), dE_median=float(np.median(dE[:, 1:])),
               slope_med={s: float(ST.loc["spec_slope", f"median_{SRC_SHORT[s]}"]) for s in SOURCES},
               slope_H=float(ST.loc["spec_slope", "H"]), slope_eps=float(ST.loc["spec_slope", "epsilon_sq"])),
          open(OUT / "signature_extra.json", "w"), indent=1)
print(ST.round(3).to_string())
