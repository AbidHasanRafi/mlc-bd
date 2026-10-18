"""Label-issue screening with confident learning (Northcutt et al., 2021) on 5-fold group-aware out-of-fold probabilities
(mean of ResNet-50 and DINOv2-S linear probes). 
 flag_cl     : argmax over classes whose probability >= that class' threshold t_k (t_k = mean p_k over images labelled k) differs from the given label
 flag_strict : flag_cl AND p(suggested) >= 0.90 AND p(given) <= 0.05
Writes outputs/label_issues.csv and outputs/label_issue_summary.json"""
import sys, pathlib, json; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, pandas as pd
from common import *
inv = pd.read_csv(OUT / "inventory.csv"); y = inv.cls.map({c: i for i, c in enumerate(CLASSES)}).values
P = np.load(OUT / "oof_ensemble.npy"); n, K = P.shape
t = np.array([P[y == k, k].mean() for k in range(K)])
cand = P >= t[None, :]                                         # classes that clear their own threshold
masked = np.where(cand, P, -1); sug = masked.argmax(1); has = cand.any(1)
flag_cl = has & (sug != y)
strict = flag_cl & (P[np.arange(n), sug] >= 0.90) & (P[np.arange(n), y] <= 0.05)
df = inv[["relpath", "source", "cls", "width", "height"]].assign(given=[CLASSES[i] for i in y], suggested=[CLASSES[i] for i in np.where(has, sug, y)],
                                                                p_given=P[np.arange(n), y], p_suggested=P[np.arange(n), np.where(has, sug, y)],
                                                                p_max=P.max(1), argmax=[CLASSES[i] for i in P.argmax(1)], flag_cl=flag_cl, flag_strict=strict)
df.to_csv(OUT / "label_issues.csv", index=False)
res = dict(thresholds=dict(zip(CLASSES, t.round(3))), oof_acc=float((P.argmax(1) == y).mean()), n_flag_cl=int(flag_cl.sum()), pct_flag_cl=float(flag_cl.mean() * 100),
           n_flag_strict=int(strict.sum()), pct_flag_strict=float(strict.mean() * 100),
           by_class_cl=df.groupby("cls").flag_cl.mean().mul(100).round(2).to_dict(), by_source_cl=df.groupby("source").flag_cl.mean().mul(100).round(2).to_dict(),
           by_class_strict=df.groupby("cls").flag_strict.sum().to_dict(), by_source_strict=df.groupby("source").flag_strict.sum().to_dict())
pairs = df[df.flag_cl].groupby(["given", "suggested"]).size().sort_values(ascending=False); res["top_pairs"] = {f"{a} -> {b}": int(v) for (a, b), v in pairs.head(12).items()}
res["top_pairs_strict"] = {f"{a} -> {b}": int(v) for (a, b), v in df[df.flag_strict].groupby(["given", "suggested"]).size().sort_values(ascending=False).head(12).items()}
json.dump(res, open(OUT / "label_issue_summary.json", "w"), indent=1); print(json.dumps(res, indent=1))
