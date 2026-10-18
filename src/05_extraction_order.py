"""Does extraction order carry site information?  Files are indexed in extraction order, so consecutive chips may come from the same site.
 1. Feature-space similarity of consecutive chips (index i, i+1) versus random same-class same-provider pairs (DINOv2-S cosine).
 2. Accuracy of a linear probe under three cross-validation designs: random stratified folds, duplicate-aware folds, and block-grouped folds
    (groups = runs of B consecutive indices within a provider x class folder).
Writes outputs/leakage_results.json and outputs/block_groups.csv"""
import sys, pathlib, json, re, warnings; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from common import *
warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)
inv = pd.read_csv(OUT / "inventory.csv"); dup = pd.read_csv(OUT / "dup_group.csv").dup_group.values
inv["idx"] = inv.file.str.extract(r"_(\d{4})\.png$")[0].astype(int)
y = inv.cls.map({c: i for i, c in enumerate(CLASSES)}).values; s = inv.source.map({c: i for i, c in enumerate(SOURCES)}).values
X = np.load(OUT / "feat_dinov2s.npy"); Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
res = {}

# 1. neighbour similarity
cons, rand = [], []
for (src, cl), g in inv.groupby(["source", "cls"]):
    g = g.sort_values("idx"); ii = g.index.values
    cons += list((Xn[ii[:-1]] * Xn[ii[1:]]).sum(1))
    a, b = rng.integers(0, len(ii), 2000), rng.integers(0, len(ii), 2000); m = a != b
    rand += list((Xn[ii[a[m]]] * Xn[ii[b[m]]]).sum(1))
from scipy import stats
res["cos_consecutive_mean"] = float(np.mean(cons)); res["cos_random_pair_mean"] = float(np.mean(rand))
res["mannwhitney_p"] = float(stats.mannwhitneyu(cons, rand, alternative="greater").pvalue)
# similarity by index distance
lag = {}
for L in [1, 2, 5, 10, 25, 50, 100, 250]:
    v = []
    for (src, cl), g in inv.groupby(["source", "cls"]):
        ii = g.sort_values("idx").index.values
        if len(ii) > L: v += list((Xn[ii[:-L]] * Xn[ii[L:]]).sum(1))
    lag[L] = float(np.mean(v))
res["cos_by_index_lag"] = lag

# 2. CV designs
def groups_block(B):
    blk = (inv.idx.values - 1) // B; gid = pd.factorize(list(zip(inv.source, inv.cls, blk)))[0]
    # merge with exact-duplicate groups
    n = len(inv); first = {}; rows, cols = [], []
    for i, (a, d) in enumerate(zip(gid, dup)):
        for key in (("b", a), ("d", d)):
            if key in first: rows.append(first[key]); cols.append(i)
            else: first[key] = i
    _, lab = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False); return lab
def cv_acc(split_iter, feat):
    pr = np.zeros(len(y), int)
    for a, b in split_iter:
        pr[b] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(feat[a], y[a]).predict(feat[b])
    return float((pr == y).mean())
strat = s * 10 + y
out = {}
for name, feat in [("dinov2s", X), ("resnet50", np.load(OUT / "feat_resnet50.npy"))]:
    out[name] = {"random": cv_acc(StratifiedKFold(5, shuffle=True, random_state=SEED).split(feat, strat), feat),
                 "dup_aware": cv_acc(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(feat, strat, dup), feat)}
    for B in [10, 25, 50, 100]:
        out[name][f"block_{B}"] = cv_acc(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(feat, strat, groups_block(B)), feat)
res["cv_accuracy"] = out
g25 = groups_block(25); pd.DataFrame(dict(relpath=inv.relpath, block_group=g25)).to_csv(OUT / "block_groups.csv", index=False)
res["n_block_groups_B25"] = int(len(np.unique(g25)))
json.dump(res, open(OUT / "leakage_results.json", "w"), indent=1); print(json.dumps(res, indent=1))
