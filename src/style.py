"""OriginPro-style figure system for IEEE/ACM double-column (7.16 in) and single-column (3.5 in) layouts.

Conventions: boxed axes (all four spines), inward major and minor ticks on all four sides, no grid, solid-filled markers with dark edges,
large axis labels (10.5 pt), tick labels 9 pt, framed legends placed inside the axes or to the right (never below), 600 dpi raster + vector PDF."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from common import *

IN = 1.0
W1, W2 = 3.5, 7.16                    # IEEE single / double column width (inches)
INK, INK2, MUTED, GRID, SURF = "#000000", "#3a3a3a", "#7f7f7f", "#e4e3de", "#ffffff"
# Provider identity = categorical slots 1-3 (validated for all-pairs use)
PROV = {"ArcGIS Map": "#2a78d6", "Bing Maps": "#eb6834", "Google Maps": "#1baf7a"}
PROV_MARK = {"ArcGIS Map": "o", "Bing Maps": "s", "Google Maps": "^"}
# Class identity: fixed order, always paired with direct labels or a legend
CLS_COL = dict(zip(CLASSES, ["#eda100", "#e87ba4", "#e34948", "#9a8c3a", "#6250d6", "#008300", "#2a78d6"]))
SEQ = ["#e8f1fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SHORT = ["Agri.", "Beach", "Built-up", "Grass.", "Hills", "Trees", "Water"]
EDGE = "#111111"
FS_LAB, FS_TICK, FS_LEG, FS_ANN, FS_PAN = 10.5, 9, 8.5, 8, 12


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": FS_TICK, "axes.titlesize": FS_LAB, "axes.labelsize": FS_LAB, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEG, "axes.linewidth": 1.1, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 4.5, "ytick.major.size": 4.5, "xtick.minor.size": 2.4, "ytick.minor.size": 2.4,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0, "xtick.minor.width": 0.7, "ytick.minor.width": 0.7,
        "xtick.major.pad": 3, "ytick.major.pad": 3, "axes.labelpad": 3,
        "axes.spines.top": True, "axes.spines.right": True, "axes.grid": False,
        "figure.dpi": 150, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
        "lines.linewidth": 1.6, "lines.markeredgewidth": 0.7, "legend.frameon": True, "legend.framealpha": 1.0,
        "legend.edgecolor": INK, "legend.fancybox": False, "legend.borderpad": 0.45, "legend.labelspacing": 0.3,
        "legend.handletextpad": 0.5, "legend.borderaxespad": 0.5, "figure.facecolor": SURF, "axes.facecolor": SURF,
        "mathtext.default": "regular"})
    matplotlib.rcParams["legend.edgecolor"] = INK


setup()


def frame(ax, minor=True):
    """Origin-like frame: inward ticks on all sides, optional minor ticks."""
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2)) if ax.get_xscale() == "linear" else None
        ax.yaxis.set_minor_locator(AutoMinorLocator(2)) if ax.get_yscale() == "linear" else None
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.1)
    return ax


def cat_axis(ax, which="y", minor=False):
    """Categorical axis: no ticks on the category side (labels only)."""
    if which == "y":
        ax.tick_params(axis="y", which="both", left=False, right=False)
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    else:
        ax.tick_params(axis="x", which="both", bottom=False, top=False)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def panel(ax, letter, dx=-0.16, dy=1.02, inside=False):
    if inside:
        ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes, fontsize=FS_PAN, fontweight="bold", va="top", ha="left")
    else:
        ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=FS_PAN, fontweight="bold", va="bottom", ha="left")


def leg(ax, loc="best", **kw):
    """Framed legend inside the axes (Origin style)."""
    kw.setdefault("fontsize", FS_LEG)
    lg = ax.legend(loc=loc, **kw)
    lg.get_frame().set_edgecolor(INK); lg.get_frame().set_linewidth(0.8)
    return lg


def mk(ax, x, y, prov, ms=6.5, **kw):
    """Solid marker with dark edge in the provider colour/shape."""
    return ax.plot(x, y, PROV_MARK[prov], color=PROV[prov], mec=EDGE, mew=0.7, ms=ms, ls="none", **kw)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig); print("saved", name)


# ------------------------------------------------------------------ helpers shared by the figure scripts
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
CMAP = LinearSegmentedColormap.from_list("mlcbd_seq", SEQ)
CMAP_DIV = LinearSegmentedColormap.from_list("mlcbd_div", ["#2a78d6", "#f7f7f5", "#eb6834"])


def heat(ax, M, vmin=None, vmax=None, fmt="{:.0f}", cmap=CMAP, thresh=None, fs=FS_ANN, mask_below=None, edge=True):
    """Annotated heat map; returns the image for a colour bar."""
    M = np.asarray(M, float)
    vmin = np.nanmin(M) if vmin is None else vmin
    vmax = np.nanmax(M) if vmax is None else vmax
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    thresh = vmin + 0.55 * (vmax - vmin) if thresh is None else thresh
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v) or (mask_below is not None and v < mask_below):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs, color="white" if v > thresh else INK)
    ax.set_xticks(range(M.shape[1])); ax.set_yticks(range(M.shape[0]))
    ax.tick_params(which="both", length=0)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.1)
    return im


def cbar(fig, ax, im, label, width="4%", pad=0.04, ticks=None):
    """Vertical colour bar to the right of an axes (inward ticks, Origin look)."""
    cax = ax.inset_axes([1.0 + pad, 0.0, 0.05, 1.0]) if width is None else ax.inset_axes([1.0 + pad, 0.0, 0.055, 1.0])
    cb = fig.colorbar(im, cax=cax, ticks=ticks)
    cb.ax.tick_params(direction="in", length=3, width=0.8, labelsize=FS_TICK - 1)
    cb.outline.set_linewidth(1.0)
    cb.set_label(label, fontsize=FS_TICK, labelpad=3)
    return cb


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def note(ax, text, loc="upper left", fs=FS_ANN, **kw):
    """Text annotation in axes coordinates (statistics, sample sizes)."""
    xy = {"upper left": (0.06, 0.97, "left", "top"), "upper right": (0.94, 0.97, "right", "top"),
          "lower left": (0.06, 0.03, "left", "bottom"), "lower right": (0.94, 0.03, "right", "bottom")}[loc]
    return ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3], fontsize=fs, linespacing=1.15, **kw)
SHORT2 = ["Agricultural", "Beaches", "Built-up", "Grasslands", "Hills", "Trees", "Water"]
