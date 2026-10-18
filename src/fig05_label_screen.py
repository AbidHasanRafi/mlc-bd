"""Fig. 5 - label screening: (a) disagreement rate with Wilson intervals, (b) assigned vs predicted, (c) the 16 flagged chips."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, pandas as pd
from PIL import Image
from matplotlib.gridspec import GridSpec
from style import *

d = pd.read_csv(OUT / "label_issues.csv")
P = np.load(OUT / "oof_ensemble.npy")
y = d.given.map({c: i for i, c in enumerate(CLASSES)}).values
d["dis"] = P.argmax(1) != y
d["pred"] = [CLASSES[i] for i in P.argmax(1)]

FH = 5.6
fig = plt.figure(figsize=(W2, FH))
L, R_ = 0.012, 0.992                      # shared left/right edge of every panel block (figure fraction)
LAB = 0.105                               # width reserved for y tick labels inside panels (a) and (b)
top_a, bot_a = 0.955, 0.50
gt = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.5, left=L + LAB, right=R_ - 0.005, top=top_a, bottom=bot_a)
gm = GridSpec(2, 8, figure=fig, hspace=0.62, wspace=0.1, left=L, right=R_, top=0.34, bottom=0.01)


def flabel(x, y_, t):
    fig.text(x, y_, t, fontsize=FS_PAN, fontweight="bold", ha="left", va="center")


# (a) disagreement rate
axa = fig.add_subplot(gt[0, 0])
yy = np.arange(7)[::-1]
for j, s in enumerate(SOURCES):
    xv, lo, hi = [], [], []
    for c in CLASSES:
        m = (d.cls == c).values & (d.source == s).values
        k, n = d.dis[m].sum(), m.sum(); l, h = wilson(k, n); xv.append(k / n * 100); lo.append(l * 100); hi.append(h * 100)
    xv, lo, hi = map(np.array, (xv, lo, hi))
    axa.errorbar(xv, yy + (1 - j) * 0.22, xerr=[xv - lo, hi - xv], fmt=PROV_MARK[s], color=PROV[s], mec=EDGE, mew=0.7, ms=5.5, elinewidth=1.3, capsize=2, label=SRC_SHORT[s])
axa.set_yticks(yy); axa.set_yticklabels(SHORT2); axa.set_xlabel("Disagreement rate (%)"); axa.set_xlim(-0.5, 11); axa.set_ylim(-0.6, 6.6)
frame(axa); cat_axis(axa, "y")
leg(axa, "upper right", fontsize=FS_ANN - 0.5, handletextpad=0.1, borderpad=0.3, labelspacing=0.2)

# (b) assigned vs predicted
axb = fig.add_subplot(gt[0, 1])
C = np.zeros((7, 7))
for g_, p_ in zip(d[d.dis].given, d[d.dis].pred):
    C[CLASSES.index(g_), CLASSES.index(p_)] += 1
heat(axb, np.where(C > 0, C, np.nan), vmin=0, vmax=C.max() * 0.95, fmt="{:.0f}", thresh=C.max() * 0.5, fs=FS_ANN)
axb.set_xticklabels(SHORT, rotation=40, ha="right", rotation_mode="anchor", fontsize=FS_TICK - 1); axb.set_yticklabels(SHORT, fontsize=FS_TICK - 1)
axb.set_xlabel("Predicted"); axb.set_ylabel("Assigned"); axb.set_title(f"{int(C.sum())} disagreements", fontsize=FS_TICK, pad=4)
fig.canvas.draw()
bb_a, bb_b = axa.get_position(), axb.get_position()
# make (a) and (b) the same height and baseline
axa.set_position([bb_a.x0, bb_b.y0, bb_a.width, bb_b.height])
flabel(L, top_a + 0.012, "(a)")
flabel(bb_b.x0 - 0.075, top_a + 0.012, "(b)")

# (c) flagged chips
f = d[d.flag_cl].sort_values("p_suggested", ascending=False).head(16)
T = 220
for k, (_, r) in enumerate(f.iterrows()):
    ax = fig.add_subplot(gm[k // 8, k % 8])
    im_ = Image.open(DATA / r.relpath).convert("RGB")
    sc = T / max(im_.size)
    im_ = im_.resize((max(1, round(im_.width * sc)), max(1, round(im_.height * sc))), Image.LANCZOS)
    canvas = Image.new("RGB", (T, T), (236, 235, 231)); canvas.paste(im_, ((T - im_.width) // 2, (T - im_.height) // 2))
    ax.imshow(canvas); ax.set_xticks([]); ax.set_yticks([]); ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.8); sp.set_edgecolor(PROV[r.source])
    g_ = SHORT[CLASSES.index(r.given)]; s_ = SHORT[CLASSES.index(r.suggested)]
    ax.set_title(f"{g_} → {s_}\n{SRC_SHORT[r.source]} {int(r.relpath.split('_')[-1][:4])}", fontsize=FS_ANN - 1.5, pad=2, linespacing=1.15)
    if k == 0:
        top_c = ax.get_position().y1
flabel(L, 0.395, "(c)")
save(fig, "Fig5_label_screen")
