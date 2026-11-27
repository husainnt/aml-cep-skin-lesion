import pandas as pd, numpy as np, os
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

# here i name every group of runs and the file pattern of its saved probabilities
groups = {
    "normal 0%":   r"outputs\missing\hyb_effb0_plain_md0.0_seed{}_0pct_hidden.npy",
    "normal 100%": r"outputs\missing\hyb_effb0_plain_md0.0_seed{}_100pct_hidden.npy",
    "dropout 0%":  r"outputs\missing\hyb_effb0_plain_md0.3_seed{}_0pct_hidden.npy",
    "dropout 100%": r"outputs\missing\hyb_effb0_plain_md0.3_seed{}_100pct_hidden.npy",
    "image only":  r"outputs\effb0_plain_aug1_seed{}_val_proba.npy",
}

def sens_at_cap(p_high, high, cap):
    # here i refer the top "cap" share of lesions and count the high concern ones inside
    order = np.argsort(-p_high)
    k = int(cap * len(order))
    return high[order][:k].sum() / high.sum()

# here i load every run and make one prediction per lesion
runs = {}
for g, pattern in groups.items():
    for seed in seeds:
        proba = np.load(os.path.join(OUT, pattern.format(seed)))
        mp, pred, labels = to_lesion_level(proba, val_lesions, val_names, classes)
        runs[(g, seed)] = {"pred": pred.values, "p_high": mp[HIGH].sum(axis=1).values}

y_true = labels.values
is_high = labels.isin(HIGH).values
n = len(y_true)

def metrics(g, idx):
    # here i calculate the measures of one group on the lesions in idx, averaged over the 3 seeds
    out = {"macro_f1": [], "mel_recall": [], "sens25": [], "auroc": []}
    for seed in seeds:
        r = runs[(g, seed)]
        yt, yp, ph, high = y_true[idx], r["pred"][idx], r["p_high"][idx], is_high[idx]
        out["macro_f1"].append(f1_score(yt, yp, average="macro"))
        out["mel_recall"].append((yp[yt == "mel"] == "mel").mean())
        out["sens25"].append(sens_at_cap(ph, high, 0.25))
        out["auroc"].append(roc_auc_score(high, ph))
    return {k: np.mean(v) for k, v in out.items()}

# here i list the comparisons: (name, group a, group b, group c, group d) means (a - b) or (a - b) - (c - d)
comparisons = [
    ("dropout minus normal, all details hidden", "dropout 100%", "normal 100%"),
    ("dropout minus normal, no details hidden", "dropout 0%", "normal 0%"),
    ("metadata effect, normal model (0% - 100% hidden)", "normal 0%", "normal 100%"),
    ("metadata effect, dropout model (0% - 100% hidden)", "dropout 0%", "dropout 100%"),
    ("dropout with details hidden minus image only", "dropout 100%", "image only"),
]

diffs = {c[0]: {k: [] for k in ["macro_f1", "mel_recall", "sens25", "auroc"]} for c in comparisons}
for b in range(500):
    idx = rng.integers(0, n, n)           # the same resampled lesions for every comparison
    cache = {g: metrics(g, idx) for g in groups}
    for name, a, c in comparisons:
        for k in diffs[name]:
            diffs[name][k].append(cache[a][k] - cache[c][k])

for name, d in diffs.items():
    print("\n" + name)
    for k, v in d.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        print("  %s: mean %.3f, 95%% CI %.3f to %.3f" % (k, np.mean(v), lo, hi))