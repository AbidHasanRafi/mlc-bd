"""Exact (SHA-256) and near-duplicate (pHash+dHash Hamming) detection within and across sources/classes.
Writes outputs/duplicate_pairs.csv and outputs/dup_group.csv (connected-component group id for split building)."""
import sys, numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
inv = pd.read_csv(OUT / "inventory.csv")
def h2u(s): return np.array([int(x, 16) for x in s], dtype=np.uint64)
P, D = h2u(inv.phash), h2u(inv.dhash)
def pop(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int16)
TH_P, TH_D = 6, 8   # conservative: both hashes must agree (phash<=6 AND dhash<=8)
ii, jj = [], []
n = len(inv)
for a in range(0, n, 500):
    hp = pop(P[a:a+500, None] ^ P[None, :]); hd = pop(D[a:a+500, None] ^ D[None, :])
    m = (hp <= TH_P) & (hd <= TH_D)
    r, c = np.nonzero(m); keep = (r + a) < c
    ii += list(r[keep] + a); jj += list(c[keep])
pairs = pd.DataFrame(dict(i=ii, j=jj))
pairs["exact"] = inv.sha256.values[pairs.i] == inv.sha256.values[pairs.j]
for k in ["source", "cls", "relpath"]:
    pairs[k + "_a"] = inv[k].values[pairs.i]; pairs[k + "_b"] = inv[k].values[pairs.j]
pairs["same_source"] = pairs.source_a == pairs.source_b; pairs["same_class"] = pairs.cls_a == pairs.cls_b
pairs.to_csv(OUT / "duplicate_pairs.csv", index=False)
g = coo_matrix((np.ones(len(pairs)), (pairs.i, pairs.j)), shape=(n, n))
nc, lab = connected_components(g, directed=False)
inv[["relpath"]].assign(dup_group=lab).to_csv(OUT / "dup_group.csv", index=False)
print("near-dup pairs:", len(pairs), "exact:", pairs.exact.sum())
print(pairs.groupby(["same_source", "same_class"]).size())
sz = pd.Series(lab).value_counts(); print("groups with >1 member:", (sz > 1).sum(), "images involved:", sz[sz > 1].sum(), "largest:", sz.max())
print(pairs[~pairs.same_class][["relpath_a", "relpath_b", "exact"]].to_string())
