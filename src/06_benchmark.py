"""Technical validation experiments on frozen embeddings (+ a handcrafted baseline).
 A. Block-grouped stratified split (70/15/15 by source x class; runs of 25 consecutive extraction indices and exact-duplicate groups kept together) -> outputs/splits.csv
 B. Within-provider vs cross-provider transfer (3x3 matrix) and leave-one-provider-out (LOPO), pooled training
 C. Out-of-fold probabilities (5-fold, group-aware) for label-issue screening (confident learning, Northcutt et al. 2021)
 D. Domain-shift quantification: provider identifiability, per-class Frechet distance between providers
Writes CSV result tables in outputs/."""
import sys, json, warnings, numpy as np, pandas as pd
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
rng = np.random.default_rng(SEED)
inv = pd.read_csv(OUT / "inventory.csv"); dupg = pd.read_csv(OUT / "dup_group.csv").dup_group.values
grp = pd.read_csv(OUT / "block_groups.csv").block_group.values   # runs of 25 consecutive extraction indices per provider x class, merged with exact-duplicate groups
y = inv.cls.map({c: i for i, c in enumerate(CLASSES)}).values; s = inv.source.map({c: i for i, c in enumerate(SOURCES)}).values
strat = (s * 10 + y)
FEATS = {k: np.load(OUT / f"feat_{k}.npy") for k in ["resnet50", "dinov2s", "handcrafted"]}
DEEP = ["resnet50", "dinov2s"]

# ---- A. splits: StratifiedGroupKFold(k=20) -> 14 folds train, 3 val, 3 test (~70/15/15), stratified by source x class
sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
fold = np.zeros(len(inv), int)
for k, (_, te) in enumerate(sgkf.split(np.zeros(len(inv)), strat, grp)): fold[te] = k
split = np.where(fold < 14, "train", np.where(fold < 17, "val", "test"))
inv_out = inv[["relpath", "source", "cls"]].assign(split=split, fold20=fold, block_group=grp, dup_group=dupg)
inv_out.to_csv(OUT / "splits.csv", index=False)
print(inv_out.groupby("split").size().to_dict())
tr, va, te = (split == "train"), (split == "val"), (split == "test")

def clf(C=1.0): return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, n_jobs=1))
def metrics(yt, yp): return dict(acc=accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average="macro"))
def boot_ci(yt, yp, n=1000):
    idx = rng.integers(0, len(yt), (n, len(yt))); a = (yt[idx] == yp[idx]).mean(1); return np.percentile(a, [2.5, 97.5])

res = []; models = {}
for fn, X in FEATS.items():
    # choose C on the pooled validation split
    best = max([0.01, 0.1, 1.0], key=lambda C: accuracy_score(y[va], clf(C).fit(X[tr], y[tr]).predict(X[va])))
    print(fn, "best C", best)
    # B1. pooled train -> pooled test
    m = clf(best).fit(X[tr], y[tr]); p = m.predict(X[te]); lo, hi = boot_ci(y[te], p)
    res.append(dict(feat=fn, setting="pooled->pooled", train="all", test="all", **metrics(y[te], p), ci_lo=lo, ci_hi=hi, n_test=te.sum()))
    # per-source test of the pooled model
    for si, sn in enumerate(SOURCES):
        mk = te & (s == si); pp = m.predict(X[mk]); lo, hi = boot_ci(y[mk], pp)
        res.append(dict(feat=fn, setting="pooled->source", train="all", test=sn, **metrics(y[mk], pp), ci_lo=lo, ci_hi=hi, n_test=mk.sum()))
    # B2. 3x3 transfer: train on all train-split images of source a, test on the test-split of source b (and *all* of b if a != b)
    for a, an in enumerate(SOURCES):
        ma = clf(best).fit(X[tr & (s == a)], y[tr & (s == a)])
        for b, bn in enumerate(SOURCES):
            mk = (te & (s == b)) if a == b else (s == b)   # cross-provider: whole target provider is unseen
            pp = ma.predict(X[mk]); lo, hi = boot_ci(y[mk], pp)
            res.append(dict(feat=fn, setting="transfer", train=an, test=bn, **metrics(y[mk], pp), ci_lo=lo, ci_hi=hi, n_test=mk.sum()))
            if a != b: models[(fn, an, bn)] = (y[mk], pp)
    # B3. leave-one-provider-out
    for b, bn in enumerate(SOURCES):
        m2 = clf(best).fit(X[s != b], y[s != b]); mk = (s == b); pp = m2.predict(X[mk]); lo, hi = boot_ci(y[mk], pp)
        res.append(dict(feat=fn, setting="LOPO", train="other two", test=bn, **metrics(y[mk], pp), ci_lo=lo, ci_hi=hi, n_test=mk.sum()))
        if fn == "dinov2s": np.save(OUT / f"lopo_cm_{SRC_SHORT[bn]}.npy", confusion_matrix(y[mk], pp, labels=range(7)))
    if fn == "dinov2s":
        np.save(OUT / "pooled_cm.npy", confusion_matrix(y[te], m.predict(X[te]), labels=range(7)))
        pd.DataFrame(m.predict_proba(X[te]), columns=CLASSES).assign(relpath=inv.relpath[te].values).to_csv(OUT / "pooled_test_proba.csv", index=False)
pd.DataFrame(res).to_csv(OUT / "benchmark_results.csv", index=False)
print(pd.DataFrame(res).round(3).to_string())

# ---- C. out-of-fold probabilities for label-issue screening (DINOv2 and ResNet50 averaged)
oof = {fn: np.zeros((len(inv), 7)) for fn in DEEP}
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
for ftr, fte in cv.split(np.zeros(len(inv)), strat, grp):
    for fn in DEEP:
        X = FEATS[fn]; oof[fn][fte] = clf(0.1).fit(X[ftr], y[ftr]).predict_proba(X[fte])
for fn in DEEP: np.save(OUT / f"oof_{fn}.npy", oof[fn]); print("OOF acc", fn, (oof[fn].argmax(1) == y).mean().round(4))
P = np.mean(list(oof.values()), axis=0); np.save(OUT / "oof_ensemble.npy", P)
print("OOF acc ensemble", (P.argmax(1) == y).mean().round(4))

# ---- D. domain shift
# D1 provider identifiability within each class: 5-fold CV accuracy of a 3-way provider classifier (chance 0.33)
dom = []
for fn, X in [(k, FEATS[k]) for k in DEEP]:
    for ci, cn in enumerate(CLASSES):
        mk = y == ci; Xc, sc, gc = X[mk], s[mk], grp[mk]
        pr = np.zeros(mk.sum(), int)
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xc, sc, gc):
            pr[b] = clf(0.1).fit(Xc[a], sc[a]).predict(Xc[b])
        dom.append(dict(feat=fn, cls=cn, provider_acc=accuracy_score(sc, pr)))
# D2 Frechet distance between provider distributions per class, on PCA-64 of the features (stabilises covariance)
def frechet(A, B):
    m1, m2 = A.mean(0), B.mean(0); c1, c2 = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    cs, _ = linalg.sqrtm(c1 @ c2, disp=False); cs = cs.real
    return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * cs))
fd = []
for fn, X in [(k, FEATS[k]) for k in DEEP]:
    Z = PCA(64, random_state=SEED).fit_transform(StandardScaler().fit_transform(X))
    for ci, cn in enumerate(CLASSES):
        for a in range(3):
            for b in range(a + 1, 3):
                fd.append(dict(feat=fn, cls=cn, pair=f"{SRC_SHORT[SOURCES[a]]}-{SRC_SHORT[SOURCES[b]]}",
                               frechet=frechet(Z[(y == ci) & (s == a)], Z[(y == ci) & (s == b)])))
    # class-vs-class reference scale: same provider, different classes
    ref = [frechet(Z[(y == i) & (s == a)], Z[(y == j) & (s == a)]) for a in range(3) for i in range(7) for j in range(i + 1, 7)]
    fd.append(dict(feat=fn, cls="__between_class_reference_median", pair="same-provider", frechet=float(np.median(ref))))
pd.DataFrame(dom).to_csv(OUT / "domain_provider_acc.csv", index=False); pd.DataFrame(fd).to_csv(OUT / "domain_frechet.csv", index=False)
print(pd.DataFrame(dom).groupby("feat").provider_acc.describe().round(3)); print(pd.DataFrame(fd).groupby(["feat", "pair"]).frechet.median().round(1))
