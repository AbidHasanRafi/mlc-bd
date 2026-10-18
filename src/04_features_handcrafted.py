"""Handcrafted colour-texture descriptor (row order == inventory.csv): 
 3x16-bin RGB histograms (48) + per-channel mean/std (6) + HSV mean/std (6) + uniform-LBP(P=8,R=1) histogram (10) + gradient-magnitude mean/std (2) = 72-d.
Computed on a 128x128 bilinear resize."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, pandas as pd, cv2
from skimage.feature import local_binary_pattern
from concurrent.futures import ThreadPoolExecutor
from common import *
inv = pd.read_csv(OUT / "inventory.csv")
def f(rel):
    im = cv2.cvtColor(cv2.imread(str(DATA / rel), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB); im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_LINEAR)
    h = np.concatenate([np.histogram(im[..., c], bins=16, range=(0, 256))[0] for c in range(3)]) / (128 * 128)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(float); rgb = im.reshape(-1, 3).astype(float)
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY); lbp = local_binary_pattern(g, 8, 1, "uniform"); lh = np.bincount(lbp.astype(int).ravel(), minlength=10)[:10] / g.size
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1); gm = np.hypot(gx, gy)
    return np.concatenate([h, rgb.mean(0), rgb.std(0), hsv.mean(0), hsv.std(0), lh, [gm.mean(), gm.std()]])
with ThreadPoolExecutor(2) as ex: X = np.array(list(ex.map(f, inv.relpath)), dtype=np.float32)
np.save(OUT / "feat_handcrafted.npy", X); print(X.shape)
