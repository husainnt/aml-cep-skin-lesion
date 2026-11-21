import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from utils import evaluate, save_result, to_lesion_level

OUT = r"D:\ML CEP\CEP_CODE"
HIGH = ["mel", "bcc", "akiec"]
rng = np.random.default_rng(42)   # here i fix the seed for the bootstrap

meta = pd.read_csv(OUT + r"\split.csv")
emb = np.load(OUT + r"\emb_resnet18.npy")
meta["sex"] = meta["sex"].replace("unknown", np.nan)
meta["localization"] = meta["localization"].replace("unknown", np.nan)

emb_cols = [f"e{i}" for i in range(emb.shape[1])]
data = pd.concat([meta[["age", "sex", "localization"]],
                  pd.DataFrame(emb, columns=emb_cols)], axis=1)

train_mask = (meta.split == "train").values
val_mask = (meta.split == "val").values
X_train, y_train = data[train_mask], meta.dx[train_mask]
X_val = data[val_mask]
y_val = meta.dx[val_mask]
val_lesions = meta.lesion_id[val_mask]

def build(kind, C):
    # same pipeline builder as before (everything is fitted on train only)
    parts = []
    if kind in ["meta", "both"]:
        parts.append(("age", Pipeline([("fill", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), ["age"]))
        parts.append(("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                      ["sex", "localization"]))
    if kind in ["image", "both"]:
        parts.append(("img", StandardScaler(), emb_cols))
    return Pipeline([("prep", ColumnTransformer(parts)),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=C))])

# here i use the best C for each model from p6
settings = {"metadata only (LR)": ("meta", 0.001),
            "image only (ResNet18+LR)": ("image", 0.01),
            "hybrid concat (ResNet18+meta+LR)": ("both", 0.01)}

os.makedirs(OUT + r"\figures", exist_ok=True)
plt.figure(figsize=(6, 5))

for name, (kind, C) in settings.items():
    model = build(kind, C)
    model.fit(X_train, y_train)
    classes = list(model.named_steps["clf"].classes_)
    proba = model.predict_proba(X_val)

    # here i turn image predictions into ONE prediction per lesion
    mean_proba, pred, labels = to_lesion_level(proba, val_lesions, y_val, classes)
    n = len(labels)
    print("\n", name, "| lesions in validation:", n)

    # here i save the lesion-level result (it replaces the image-level row of the same name)
    result = evaluate(name, labels, pred)
    result["best_C"] = C
    save_result(result)

    # ---- bootstrap: every row is a lesion now, so i resample rows directly ----
    is_high = labels.isin(HIGH).values
    referred_all = pred.isin(HIGH).values
    f1s, sens, rates = [], [], []
    for b in range(1000):
        idx = rng.integers(0, n, n)
        f1s.append(f1_score(labels.values[idx], pred.values[idx], average="macro"))
        high = is_high[idx]
        sens.append(referred_all[idx][high].mean())
        rates.append(referred_all[idx].mean())
    print("macro F1: %.3f (95%% CI %.3f to %.3f)" % (np.mean(f1s), *np.percentile(f1s, [2.5, 97.5])))
    print("referral sensitivity: %.3f (95%% CI %.3f to %.3f)" % (np.mean(sens), *np.percentile(sens, [2.5, 97.5])))
    print("referral rate: %.3f (95%% CI %.3f to %.3f)" % (np.mean(rates), *np.percentile(rates, [2.5, 97.5])))

    # ---- referral curve: sort lesions by probability of high concern ----
    p_high = mean_proba[HIGH].sum(axis=1).values
    order = np.argsort(-p_high)
    cum_high = np.cumsum(is_high[order]) / is_high.sum()
    ref_rate = np.arange(1, n + 1) / n
    plt.plot(ref_rate, cum_high, label=name)
    for cap in [0.20, 0.25, 0.30]:
        k = int(cap * n)
        print("  at referral rate %.2f -> sensitivity %.3f" % (cap, cum_high[k - 1]))

# here i draw the diagonal = referring lesions at random
plt.plot([0, 1], [0, 1], "k--", label="random referral")
plt.xlabel("referral rate (lesions)")
plt.ylabel("referral sensitivity")
plt.title("Referral curve, lesion level (validation set)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(OUT + r"\figures\fig_referral_curve.png", dpi=200, bbox_inches="tight")
print("\nfigure saved")