"""Per-chip radial power spectrum (native pixels) and CIELAB colour.
Spectrum: grey image, central 64 x 64 px crop at native resolution (chips with a shorter side < 64 px are skipped), Hann window,
2-D FFT, azimuthal average over integer radius r = 1..31 (frequency f = r / 64 cycles per pixel). Slope = least-squares exponent of
log10 P vs log10 f over r = 3..28.
Colour: CIELAB (D65) of the mean RGB of each chip.
Writes outputs/psd_abs.npy, psd_rel.npy (N x 31), spectral.csv (slope, valid) and lab.csv (L, a, b, chroma, hue)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, pandas as pd, cv2
from skimage.color import rgb2lab
from concurrent.futures import ThreadPoolExecutor
from common import *

inv = pd.read_csv(OUT / "inventory.csv")
N = 64
win = np.outer(np.hanning(N), np.hanning(N)).astype(np.float64)
yy, xx = np.indices((N, N)); rr = np.rint(np.hypot(yy - N // 2, xx - N // 2)).astype(int)
R = np.arange(1, 32)
norm = (win ** 2).sum()


def spec(rel):
    im = cv2.imread(str(DATA / rel), cv2.IMREAD_COLOR)
    h, w = im.shape[:2]
    if min(h, w) < N:
        return np.full(31, np.nan)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float64)
    y0, x0 = (h - N) // 2, (w - N) // 2
    c = g[y0:y0 + N, x0:x0 + N]
    c = (c - c.mean()) * win
    P = np.abs(np.fft.fftshift(np.fft.fft2(c))) ** 2 / norm
    return np.array([P[rr == r].mean() for r in R])


with ThreadPoolExecutor(4) as ex:
    P = np.array(list(ex.map(spec, inv.relpath)))
valid = ~np.isnan(P[:, 0]) & (P[:, 3:28] > 0).all(axis=1)
Prel = P / np.nansum(P, axis=1, keepdims=True)
f = R / N
sel = (R >= 3) & (R <= 28)
slope = np.full(len(inv), np.nan)
lf = np.log10(f[sel])
for i in np.where(valid)[0]:
    slope[i] = np.polyfit(lf, np.log10(P[i, sel]), 1)[0]
np.save(OUT / "psd_abs.npy", P); np.save(OUT / "psd_rel.npy", Prel)
pd.DataFrame(dict(relpath=inv.relpath, spec_valid=valid, spec_slope=slope)).to_csv(OUT / "spectral.csv", index=False)

rgb = inv[["mean_r", "mean_g", "mean_b"]].values / 255.0
lab = rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)
chroma = np.hypot(lab[:, 1], lab[:, 2]); hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360
pd.DataFrame(dict(relpath=inv.relpath, L=lab[:, 0], a=lab[:, 1], b=lab[:, 2], chroma=chroma, hue=hue)).to_csv(OUT / "lab.csv", index=False)
d = pd.DataFrame(dict(source=inv.source, slope=slope, valid=valid))
print("valid:", int(valid.sum()))
print(d[d.valid].groupby("source").slope.describe().round(3))
