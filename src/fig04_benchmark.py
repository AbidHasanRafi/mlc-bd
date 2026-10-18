"""Fig. 4 - baseline benchmark, cross-provider transfer, adaptation and learning curves."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, pandas as pd
from style import *

B = pd.read_csv(OUT / "benchmark_results.csv")
ad = pd.read_csv(OUT / "adaptation_results.csv")
apc = pd.read_csv(OUT / "adaptation_perclass.csv")
lc = pd.read_csv(OUT / "learning_curves.csv")
fig, axs = plt.subplots(2, 3, figsize=(W2, 6.4), gridspec_kw=dict(hspace=0.5, wspace=0.85, left=0.085, right=0.985, top=0.955, bottom=0.09))

# (a) transfer matrix, DINOv2-S
ax = axs[0, 0]; panel(ax, "a", dx=-0.38)
T = B[(B.feat == "dinov2s") & (B.setting == "transfer")]
M = np.array([[T[(T.train == a) & (T.test == b)].acc.values[0] * 100 for b in SOURCES] for a in SOURCES])
im = heat(ax, M, vmin=60, vmax=100, fmt="{:.1f}", thresh=82, fs=FS_ANN + 0.5)
for i in range(3):
    ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=INK, lw=1.8))
ax.set_xticklabels([SRC_SHORT[s] for s in SOURCES]); ax.set_yticklabels([SRC_SHORT[s] for s in SOURCES])
ax.set_xlabel("Test provider"); ax.set_ylabel("Training provider")
ax.set_title("Accuracy (%), DINOv2-S", fontsize=FS_TICK, pad=4)
cbar(fig, ax, im, "")

# (b) protocols x feature sets
ax = axs[0, 1]; panel(ax, "b", dx=-0.62)
feats = [("handcrafted", "Color–texture", "^", "#9a9a9a"), ("resnet50", "ResNet-50", "s", "#5a5a5a"), ("dinov2s", "DINOv2-S", "o", INK)]
rows = ["Pooled", "Within\nprovider", "Leave one\nprovider out", "One provider\nto another"]


def cells(f, k):
    d = B[B.feat == f]
    if k == 0: return d[d.setting == "pooled->pooled"].macro_f1.values * 100
    if k == 1: return d[(d.setting == "transfer") & (d.train == d.test)].macro_f1.values * 100
    if k == 2: return d[d.setting == "LOPO"].macro_f1.values * 100
    return d[(d.setting == "transfer") & (d.train != d.test)].macro_f1.values * 100


yy = np.arange(4)[::-1]
off = {"handcrafted": 0.22, "resnet50": 0.0, "dinov2s": -0.22}
for f, nm, mkr, col in feats:
    for k in range(4):
        v = cells(f, k); y_ = yy[k] + off[f]
        ax.plot([v.min(), v.max()], [y_, y_], color=col, lw=1.6, zorder=2, solid_capstyle="butt")
        ax.plot(v.mean(), y_, mkr, color=col, mec=EDGE, mew=0.8, ms=7, zorder=3, label=nm if k == 0 else None)
ax.set_yticks(yy); ax.set_yticklabels(rows); ax.set_xlabel("Macro-F1 (%)"); ax.set_xlim(20, 102); ax.set_ylim(-0.6, 3.6)
frame(ax); cat_axis(ax, "y")
leg(ax, "upper left", fontsize=FS_ANN - 0.5, handletextpad=0.1, borderpad=0.3)

# (c) per-class recall
ax = axs[0, 2]; panel(ax, "c", dx=-0.45)
lo = {s: np.load(OUT / f"lopo_cm_{SRC_SHORT[s]}.npy") for s in SOURCES}
pc = np.load(OUT / "pooled_cm.npy")
R = np.array([pc.diagonal() / pc.sum(1)] + [lo[s].diagonal() / lo[s].sum(1) for s in SOURCES]).T * 100
im = heat(ax, R, vmin=30, vmax=100, fmt="{:.0f}", thresh=70, fs=FS_ANN)
ax.set_xticklabels(["Pooled", "−ArcGIS", "−Bing", "−Google"], rotation=30, ha="right", rotation_mode="anchor", fontsize=FS_TICK - 1)
ax.set_yticklabels(SHORT2); ax.set_title("Recall (%), DINOv2-S", fontsize=FS_TICK, pad=4)
cbar(fig, ax, im, "")

# (d) adaptation baselines
ax = axs[1, 0]; panel(ax, "d", dx=-0.3)
methods = ["none", "zscore", "coral"]; mlab = ["None", "Z-score", "CORAL"]
d_ = ad[ad.feat == "dinov2s"]
for s in SOURCES:
    v = [d_[(d_.method == m) & (d_.target == s)].acc.values[0] * 100 for m in methods]
    ax.plot(range(3), v, "-", color=PROV[s], lw=1.6, zorder=2)
    ax.plot(range(3), v, PROV_MARK[s], color=PROV[s], mec=EDGE, mew=0.8, ms=7, ls="none", label=SRC_SHORT[s] + " unseen", zorder=3)
mean = [d_[d_.method == m].acc.mean() * 100 for m in methods]
ax.plot(range(3), mean, color=INK, lw=2.2, ls="--", zorder=1, label="Mean")
ax.set_xticks(range(3)); ax.set_xticklabels(mlab, fontsize=FS_TICK - 1); ax.set_xlim(-0.4, 2.4)
ax.set_ylabel("LOPO accuracy (%)"); ax.set_ylim(55, 95); frame(ax, minor=False); ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
ax.tick_params(axis="x", which="both", bottom=False, top=False)
leg(ax, "lower left", fontsize=FS_ANN - 1, handletextpad=0.1, borderpad=0.3, labelspacing=0.2)

# (e) learning curves
ax = axs[1, 1]; panel(ax, "e", dx=-0.3)
sty = {"Pooled": (INK, "o", "-"), "Within provider": ("#5a5a5a", "s", "--"), "One provider to another": ("#9a8c3a", "^", "-"), "Leave one provider out": ("#6250d6", "D", "-")}
for p, (col, mkr, ls) in sty.items():
    g = lc[lc.protocol == p].groupby("frac").acc.agg(["mean", "std"]) * 100
    ax.errorbar(g.index, g["mean"], yerr=g["std"], color=col, marker=mkr, ls=ls, lw=1.6, ms=5.5, mec=EDGE, mew=0.6, capsize=2, label=p)
ax.set_xscale("log"); ax.set_xticks([0.02, 0.05, 0.1, 0.25, 0.5, 1]); ax.set_xticklabels(["2", "5", "10", "25", "50", "100"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("Training data used (%)"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(62, 100); frame(ax, minor=False)
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
leg(ax, "lower right", fontsize=FS_ANN - 1.5, handletextpad=0.2, borderpad=0.3, labelspacing=0.2, handlelength=1.8)

# (f) per-class effect of z-scoring, Google Maps unseen
ax = axs[1, 2]; panel(ax, "f", dx=-0.45)
g = apc[apc.target == "Google Maps"].pivot_table(index="cls", columns="method", values="recall").loc[CLASSES] * 100
yy7 = np.arange(7)[::-1]
for i, c in zip(yy7, CLASSES):
    ax.annotate("", xy=(g.loc[c, "zscore"], i), xytext=(g.loc[c, "none"], i), arrowprops=dict(arrowstyle="-|>", color="#7f7f7f", lw=1.3, shrinkA=4, shrinkB=4))
ax.plot(g["none"].values, yy7, "o", color="white", mec=INK, mew=1.5, ms=7, ls="none", label="None", zorder=3)
ax.plot(g["zscore"].values, yy7, "o", color=INK, mec=INK, ms=7, ls="none", label="Z-score", zorder=3)
ax.set_yticks(yy7); ax.set_yticklabels(SHORT2); ax.set_xlabel("Recall on Google Maps (%)"); ax.set_xlim(25, 102)
frame(ax); cat_axis(ax, "y")
leg(ax, "lower left", fontsize=FS_ANN - 0.5, handletextpad=0.1, borderpad=0.3, labelspacing=0.2)
save(fig, "Fig4_benchmark")
