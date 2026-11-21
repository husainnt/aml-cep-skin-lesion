import pandas as pd, numpy as np, os
from sklearn.metrics import roc_auc_score, f1_score
from utils import to_lesion_level

OUT = r"D:\ML CEP\CEP_CODE"
HIGH = ["mel", "bcc", "akiec"]
rng = np.random.default_rng(42)

# here i list the runs that i want to compare (the file names come from cell B)
runs = {"plain": "effb0_plain_aug1_seed42",
        "weighted": "effb0_weighted_aug1_seed42",
        "focal": "effb0_focal_aug1_seed42"}

meta = pd.read_csv(OUT + r"\split.csv")
val_mask = (meta.split == "val").values
val_lesions = meta.lesion_id[val_mask]
val_names = meta.dx[val_mask]
classes = sorted(meta.dx.unique())

def sens_at_cap(p_high, high, cap):
    # here i refer the top "cap" share of lesions and count the high concern ones inside
    order = np.argsort(-p_high)
    k = int(cap * len(order))
    return high[order][:k].sum() / high.sum()

# here i load every run and make one probability vector per lesion
store = {}
for name, fname in runs.items():
    path = OUT + r"\outputs\{}_val_proba.npy".format(fname)
    if not os.path.exists(path):
        print("skipping", name, "(file not found)")
        continue
    proba = np.load(path)
    mean_proba, pred, labels = to_lesion_level(proba, val_lesions, val_names, classes)
    store[name] = {"pred": pred.values, "p_high": mean_proba[HIGH].sum(axis=1).values}

y_true = labels.values
is_high = labels.isin(HIGH).values
n = len(y_true)
print("lesions in validation:", n, "| high concern lesions:", is_high.sum())

# here i print the table: same referral caps for every run
print("\nrun | macroF1 | mel recall | AUROC(p_high) | sens@20% | sens@25% | sens@30%")
for name, s in store.items():
    f1 = f1_score(y_true, s["pred"], average="macro")
    mel_rec = (s["pred"][y_true == "mel"] == "mel").mean()
    auc = roc_auc_score(is_high, s["p_high"])
    caps = [sens_at_cap(s["p_high"], is_high, c) for c in [0.20, 0.25, 0.30]]
    print("%s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f" % (name, f1, mel_rec, auc, *caps))

# here i do the paired bootstrap against "plain": the same resampled lesions for both models
for name in store:
    if name == "plain":
        continue
    d_s25, d_auc = [], []
    for b in range(1000):
        idx = rng.integers(0, n, n)
        d_s25.append(sens_at_cap(store[name]["p_high"][idx], is_high[idx], 0.25)
                     - sens_at_cap(store["plain"]["p_high"][idx], is_high[idx], 0.25))
        d_auc.append(roc_auc_score(is_high[idx], store[name]["p_high"][idx])
                     - roc_auc_score(is_high[idx], store["plain"]["p_high"][idx]))
    for label, d in [("sens@25%", d_s25), ("AUROC", d_auc)]:
        lo, hi = np.percentile(d, [2.5, 97.5])
        print("%s - plain, %s: mean %.3f, 95%% CI %.3f to %.3f" % (name, label, np.mean(d), lo, hi))