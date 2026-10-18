"""Per-image audit: decode check, geometry, alpha channel, SHA-256, perceptual hashes,
colour and sharpness statistics. Writes outputs/inventory.csv."""
import hashlib, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, cv2, imagehash
from PIL import Image
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *

def audit(item):
    src, cls, p = item
    raw = p.read_bytes()
    rec = dict(source=src, cls=cls, file=p.name, relpath=f"{src}/{cls}/{p.name}",
               bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    try:
        im = Image.open(p); im.load()
    except Exception as e:
        rec.update(decode_ok=False, error=str(e)); return rec
    rec.update(decode_ok=True, mode=im.mode, width=im.width, height=im.height)
    a = np.asarray(im)
    if im.mode == "RGBA":
        rec["alpha_min"] = int(a[..., 3].min()); rec["alpha_frac_nonopaque"] = float((a[..., 3] < 255).mean())
        rgb = a[..., :3]
    elif im.mode == "RGB":
        rec["alpha_min"] = 255; rec["alpha_frac_nonopaque"] = 0.0; rgb = a
    else:
        rgb = np.asarray(im.convert("RGB")); rec["alpha_min"] = 255; rec["alpha_frac_nonopaque"] = 0.0
    rgb = np.ascontiguousarray(rgb)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rec.update(mean_r=rgb[..., 0].mean(), mean_g=rgb[..., 1].mean(), mean_b=rgb[..., 2].mean(),
               std_gray=float(g.std()))
    rec["brightness"] = (rec["mean_r"] + rec["mean_g"] + rec["mean_b"]) / 3
    # sharpness: variance of Laplacian on a fixed 128 px short-side-independent resize so sizes are comparable
    g128 = cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA)
    rec["lap_var_native"] = float(cv2.Laplacian(g, cv2.CV_64F).var()) if min(g.shape) >= 3 else np.nan
    rec["lap_var_128"] = float(cv2.Laplacian(g128, cv2.CV_64F).var())
    # fraction of spectral energy above half of Nyquist (upsampling/blur indicator) on the native image
    if min(g.shape) >= 16:
        c = cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA) if max(g.shape) > 128 else g
        F = np.abs(np.fft.fftshift(np.fft.fft2(c.astype(np.float32) - c.mean()))) ** 2
        yy, xx = np.indices(F.shape); r = np.hypot(yy - F.shape[0] / 2, xx - F.shape[1] / 2) / (min(F.shape) / 2)
        rec["hf_energy_ratio"] = float(F[r > 0.5].sum() / (F.sum() + 1e-9))
    hist = np.bincount(g.ravel(), minlength=256) / g.size
    rec["entropy"] = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())
    pil = Image.fromarray(rgb)
    rec["phash"] = str(imagehash.phash(pil, hash_size=8)); rec["dhash"] = str(imagehash.dhash(pil, hash_size=8))
    return rec

items = [(s, c, p) for s in SOURCES for c in CLASSES for p in sorted((DATA / s / c).glob("*"))]
print(len(items), "files")
with ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(audit, items))
df = pd.DataFrame(rows)
df["pixels"] = df.width * df.height
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "inventory.csv", index=False)
print(df.decode_ok.value_counts(), df["mode"].value_counts(), "total MB", df.bytes.sum() / 2**20)
