import pandas as pd, numpy as np, os
from sklearn.metrics import roc_auc_score, f1_score
from utils import to_lesion_level

OUT = r"D:\ML CEP\CEP_CODE"
HIGH = ["mel", "bcc", "akiec"]
rng = np.random.default_rng(42)

losses = ["plain", "weighted", "focal"]
seeds = [42, 1, 2]

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

# here i load all runs and make one prediction per lesion
runs = {}
for loss in losses:
    for seed in seeds:
        name = "effb0_{}_aug1_seed{}".format(loss, seed)
        path = OUT + r"\outputs\{}_val_proba.npy".format(name)
        if not os.path.exists(path):
            print("missing:", name)
            continue
        proba = np.load(path)
        mean_proba, pred, labels = to_lesion_level(proba, val_lesions, val_names, classes)
        runs[(loss, seed)] = {"pred": pred.values, "p_high": mean_proba[HIGH].sum(axis=1).values}

y_true = labels.values
is_high = labels.isin(HIGH).values
n = len(y_true)
print("lesions:", n, "| high concern:", is_high.sum(), "| runs loaded:", len(runs))

# here i calculate the numbers of every single run
rows = []
for (loss, seed), r in runs.items():
    rows.append({"loss": loss, "seed": seed,
                 "macro_f1": f1_score(y_true, r["pred"], average="macro"),
                 "mel_recall": (r["pred"][y_true == "mel"] == "mel").mean(),
                 "auroc": roc_auc_score(is_high, r["p_high"]),
                 "sens20": sens_at_cap(r["p_high"], is_high, 0.20),
                 "sens25": sens_at_cap(r["p_high"], is_high, 0.25),
                 "sens30": sens_at_cap(r["p_high"], is_high, 0.30)})
df = pd.DataFrame(rows)
print("\nevery run:\n", df.round(3).to_string(index=False))

# here i print mean and standard deviation over the seeds for each loss
print("\nmean over seeds:\n", df.groupby("loss").mean(numeric_only=True).drop(columns="seed").round(3))
print("\nstd over seeds:\n", df.groupby("loss").std(numeric_only=True).drop(columns="seed").round(3))

# here i do the paired bootstrap: same resampled lesions for all losses
# for every resample i average the metric over the 3 seeds of a loss, then take the difference
def mean_metric(loss, idx, which):
    vals = []
    for seed in seeds:
        r = runs[(loss, seed)]
        if which == "sens25":
            vals.append(sens_at_cap(r["p_high"][idx], is_high[idx], 0.25))
        else:
            vals.append(roc_auc_score(is_high[idx], r["p_high"][idx]))
    return np.mean(vals)

pairs = [("weighted", "plain"), ("focal", "plain"), ("focal", "weighted")]
diffs = {(a, b, w): [] for a, b in pairs for w in ["sens25", "auroc"]}
for b_i in range(500):
    idx = rng.integers(0, n, n)
    for a, b in pairs:
        for w in ["sens25", "auroc"]:
            diffs[(a, b, w)].append(mean_metric(a, idx, w) - mean_metric(b, idx, w))

print("\npaired bootstrap (difference of the 3-seed average):")
for (a, b, w), d in diffs.items():
    lo, hi = np.percentile(d, [2.5, 97.5])
    print("%s - %s, %s: mean %.3f, 95%% CI %.3f to %.3f" % (a, b, w, np.mean(d), lo, hi))