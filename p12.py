import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from utils import to_lesion_level

OUT = r"D:\ML CEP\CEP_CODE"
HIGH = ["mel", "bcc", "akiec"]
rng = np.random.default_rng(42)
seeds = [42, 1, 2]

meta = pd.read_csv(OUT + r"\split.csv")
val_mask = (meta.split == "val").values
val_lesions = meta.lesion_id[val_mask]
val_names = meta.dx[val_mask]
classes = sorted(meta.dx.unique())

# here i name the two groups of runs that i compare (both use plain loss)
groups = {"image only": "effb0_plain_aug1_seed{}",
          "hybrid": "hyb_effb0_plain_md0.0_seed{}"}

def sens_at_cap(p_high, high, cap):
    # here i refer the top "cap" share of lesions and count the high concern ones inside
    order = np.argsort(-p_high)
    k = int(cap * len(order))
    return high[order][:k].sum() / high.sum()

# here i load every run and make one prediction per lesion
runs = {}
for g, pattern in groups.items():
    for seed in seeds:
        proba = np.load(OUT + r"\outputs\{}_val_proba.npy".format(pattern.format(seed)))
        mp, pred, labels = to_lesion_level(proba, val_lesions, val_names, classes)
        runs[(g, seed)] = {"pred": pred.values, "p_high": mp[HIGH].sum(axis=1).values}

y_true = labels.values
is_high = labels.isin(HIGH).values
n = len(y_true)

def metrics(g, idx):
    # here i calculate the measures for one group on the lesions in idx and average over the 3 seeds
    out = {"macro_f1": [], "mel_recall": [], "ref_sens": [], "ref_rate": [], "sens25": [], "auroc": []}
    for seed in seeds:
        r = runs[(g, seed)]
        yt, yp, ph, high = y_true[idx], r["pred"][idx], r["p_high"][idx], is_high[idx]
        referred = np.isin(yp, HIGH)
        out["macro_f1"].append(f1_score(yt, yp, average="macro"))
        out["mel_recall"].append((yp[yt == "mel"] == "mel").mean())
        out["ref_sens"].append(referred[high].mean())
        out["ref_rate"].append(referred.mean())
        out["sens25"].append(sens_at_cap(ph, high, 0.25))
        out["auroc"].append(roc_auc_score(high, ph))
    return {k: np.mean(v) for k, v in out.items()}

# here i print the numbers on the full validation set
everything = np.arange(n)
print("measures on the full validation set (3-seed mean):")
print(pd.DataFrame({g: metrics(g, everything) for g in groups}).round(3))

# here i do the paired bootstrap: the same resampled lesions are used for both groups
diffs = {k: [] for k in ["macro_f1", "mel_recall", "ref_sens", "ref_rate", "sens25", "auroc"]}
for b in range(500):
    idx = rng.integers(0, n, n)
    a = metrics("hybrid", idx)
    c = metrics("image only", idx)
    for k in diffs:
        diffs[k].append(a[k] - c[k])

print("\npaired bootstrap, hybrid minus image only:")
for k, d in diffs.items():
    lo, hi = np.percentile(d, [2.5, 97.5])
    print("%s: mean %.3f, 95%% CI %.3f to %.3f" % (k, np.mean(d), lo, hi))

# here i print the recall of every class (3-seed mean) for both groups
print("\nper-class recall:")
table = {}
for g in groups:
    per_class = []
    for seed in seeds:
        yp = runs[(g, seed)]["pred"]
        per_class.append([(yp[y_true == c] == c).mean() for c in classes])
    table[g] = np.mean(per_class, axis=0)
out = pd.DataFrame(table, index=classes).round(3)
out["n_lesions"] = [(y_true == c).sum() for c in classes]
print(out)