"""Embedding geometry, provider identification, adaptation baselines and learning curves (frozen features, block-grouped splits).
 1. Silhouette of class structure (class labels) and of provider structure within each class (provider labels); DINOv2-S and ResNet-50.
 2. Class-by-provider centroids in the first two principal components (domain-shift vectors).
 3. Out-of-fold provider identification (5 block-grouped folds): confusion matrix.
 4. Leave-one-provider-out accuracy with two unsupervised adaptation baselines that use the unlabelled target features:
      zscore : per-provider standardisation of every feature
      coral  : per-provider standardisation, then re-colouring of each source provider with the target covariance (CORAL, Sun et al. 2016)
 5. Learning curves: accuracy against the fraction of training chips under four protocols.
Writes outputs/silhouette.csv, centroids.csv, provider_cm.npy, adaptation_results.csv, adaptation_perclass.csv, learning_curves.csv"""
import sys, pathlib, warnings; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_samples, accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from common import *
warnings.filterwarnings("ignore")

inv = pd.read_csv(OUT / "inventory.csv")
sp = pd.read_csv(OUT / "splits.csv")
grp = pd.read_csv(OUT / "block_groups.csv").block_group.values
y = inv.cls.map({c: i for i, c in enumerate(CLASSES)}).values
s = inv.source.map({c: i for i, c in enumerate(SOURCES)}).values
split = sp.split.values
rng = np.random.default_rng(SEED)
FE = {k: np.load(OUT / f"feat_{k}.npy") for k in ["dinov2s", "resnet50", "handcrafted"]}


def lr(C=0.1):
    return LogisticRegression(C=C, max_iter=3000)


# ---------------------------------------------------------------- 1-2 silhouette, centroids
rows, cent = [], []
for fn in ["dinov2s", "resnet50"]:
    Z = PCA(50, random_state=SEED).fit_transform(StandardScaler().fit_transform(FE[fn]))
    sc = silhouette_samples(Z, y)
    for i, c in enumerate(CLASSES):
        rows.append(dict(feat=fn, kind="class", cls=c, silhouette=sc[y == i].mean()))
        m = y == i
        sp_ = silhouette_samples(Z[m], s[m])
        rows.append(dict(feat=fn, kind="provider_within_class", cls=c, silhouette=sp_.mean()))
    rows.append(dict(feat=fn, kind="class", cls="ALL", silhouette=sc.mean()))
    if fn == "dinov2s":
        Z2 = PCA(2, random_state=SEED).fit(StandardScaler().fit_transform(FE[fn]))
        P2 = Z2.transform(StandardScaler().fit_transform(FE[fn]))
        for i, c in enumerate(CLASSES):
            for j, sname in enumerate(SOURCES):
                m = (y == i) & (s == j)
                cent.append(dict(cls=c, source=sname, pc1=P2[m, 0].mean(), pc2=P2[m, 1].mean(), n=int(m.sum())))
pd.DataFrame(rows).to_csv(OUT / "silhouette.csv", index=False)

# centroid-distance structure of the 21 class x provider groups (PCA-50 of DINOv2-S features), quoted in the article
import json
Xc = PCA(50, random_state=SEED).fit_transform(StandardScaler().fit_transform(FE["dinov2s"]))
groups = [(c, p) for c in CLASSES for p in SOURCES]
Cg = np.array([Xc[((inv.cls == c).values & (inv.source == p).values)].mean(0) for c, p in groups])
Dg = np.linalg.norm(Cg[:, None] - Cg[None], axis=2)
same_c = np.array([[a[0] == b[0] for b in groups] for a in groups]); same_p = np.array([[a[1] == b[1] for b in groups] for a in groups])
nn = np.where(~np.eye(21, dtype=bool), Dg, np.inf).argmin(1)
json.dump(dict(d_cross_provider=float(Dg[same_c & ~same_p].mean()), d_cross_class=float(Dg[~same_c & same_p].mean()),
               nn_same_class=float(np.mean([groups[nn[i]][0] == groups[i][0] for i in range(21)])),
               nn_same_provider=float(np.mean([groups[nn[i]][1] == groups[i][1] for i in range(21)]))),
          open(OUT / "centroid_structure.json", "w"), indent=1)

pd.DataFrame(cent).to_csv(OUT / "centroids.csv", index=False)
print(pd.DataFrame(rows).pivot_table(index="cls", columns=["feat", "kind"], values="silhouette").round(3))

# ---------------------------------------------------------------- 3 provider identification
X = StandardScaler().fit_transform(FE["dinov2s"])
pr = np.zeros(len(s), int)
for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, s * 10 + y, grp):
    pr[b] = lr().fit(X[a], s[a]).predict(X[b])
cm = confusion_matrix(s, pr)
np.save(OUT / "provider_cm.npy", cm)
print("provider ID acc:", (pr == s).mean().round(4)); print(cm)

# ---------------------------------------------------------------- 4 adaptation baselines (LOPO)
def mean_std(Z):
    return Z.mean(0), Z.std(0) + 1e-6


def sqrtm_psd(C, power):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-8, None)
    return (V * (w ** power)) @ V.T


def adapt(Z, target, method):
    """Return features (n x d) after adaptation for every provider, given the held-out target provider."""
    out = np.zeros_like(Z)
    for p in range(3):
        m = s == p
        mu, sd = mean_std(Z[m])
        out[m] = (Z[m] - mu) / sd
    if method == "coral":
        Ct = np.cov(out[s == target], rowvar=False) + 0.1 * np.eye(Z.shape[1])
        Ct_h = sqrtm_psd(Ct, 0.5)
        for p in range(3):
            if p == target:
                continue
            Cs = np.cov(out[s == p], rowvar=False) + 0.1 * np.eye(Z.shape[1])
            out[s == p] = out[s == p] @ sqrtm_psd(Cs, -0.5) @ Ct_h
    return out


res, perclass = [], []
for fn in ["dinov2s", "resnet50", "handcrafted"]:
    Z0 = StandardScaler().fit_transform(FE[fn])
    Z = PCA(128, random_state=SEED).fit_transform(Z0) if fn != "handcrafted" else Z0
    for t, tname in enumerate(SOURCES):
        tr = s != t; te = s == t
        for method in ["none", "zscore", "coral"]:
            if method == "none":
                A = Z
            else:
                A = adapt(Z, t, method)
            clf = lr().fit(A[tr], y[tr]); p = clf.predict(A[te])
            res.append(dict(feat=fn, method=method, target=tname, acc=accuracy_score(y[te], p), macro_f1=f1_score(y[te], p, average="macro")))
            if fn == "dinov2s":
                rc = recall_score(y[te], p, average=None, labels=range(7))
                for i, c in enumerate(CLASSES):
                    perclass.append(dict(method=method, target=tname, cls=c, recall=rc[i]))
R_ = pd.DataFrame(res); R_.to_csv(OUT / "adaptation_results.csv", index=False)
pd.DataFrame(perclass).to_csv(OUT / "adaptation_perclass.csv", index=False)
print(R_.pivot_table(index=["feat", "method"], columns="target", values="acc").round(3))

# ---------------------------------------------------------------- 5 learning curves (DINOv2-S)
X = StandardScaler().fit_transform(FE["dinov2s"])
trm, tem = split == "train", split == "test"
fracs = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
lc = []


def subsample(mask, frac, seed):
    r = np.random.default_rng(seed)
    idx = np.where(mask)[0]
    keep = []
    for key in np.unique(s[idx] * 10 + y[idx]):
        g = idx[(s[idx] * 10 + y[idx]) == key]
        n = max(2, int(round(len(g) * frac)))
        keep += list(r.choice(g, min(n, len(g)), replace=False))
    return np.array(keep)


for frac in fracs:
    for rep in range(5):
        seed = SEED + rep
        # pooled
        idx = subsample(trm, frac, seed)
        p = lr().fit(X[idx], y[idx]).predict(X[tem])
        lc.append(dict(frac=frac, rep=rep, protocol="Pooled", acc=accuracy_score(y[tem], p), n_train=len(idx)))
        w, c_, l_ = [], [], []
        for a in range(3):
            idx = subsample(trm & (s == a), frac, seed)
            mdl = lr().fit(X[idx], y[idx])
            idx_all = subsample(s == a, frac, seed)
            mdl_all = lr().fit(X[idx_all], y[idx_all])
            w.append(accuracy_score(y[tem & (s == a)], mdl.predict(X[tem & (s == a)])))
            for b in range(3):
                if b != a:
                    c_.append(accuracy_score(y[s == b], mdl_all.predict(X[s == b])))
        lc.append(dict(frac=frac, rep=rep, protocol="Within provider", acc=np.mean(w), n_train=int(frac * trm.sum() / 3)))
        lc.append(dict(frac=frac, rep=rep, protocol="One provider to another", acc=np.mean(c_), n_train=int(frac * trm.sum() / 3)))
        for t in range(3):
            idx = subsample(s != t, frac, seed)
            l_.append(accuracy_score(y[s == t], lr().fit(X[idx], y[idx]).predict(X[s == t])))
        lc.append(dict(frac=frac, rep=rep, protocol="Leave one provider out", acc=np.mean(l_), n_train=int(frac * trm.sum() * 2 / 3)))
L_ = pd.DataFrame(lc); L_.to_csv(OUT / "learning_curves.csv", index=False)
print(L_.groupby(["protocol", "frac"]).acc.mean().unstack().round(3))
