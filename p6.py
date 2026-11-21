import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from utils import evaluate, save_result

OUT = r"D:\ML CEP\CEP_CODE"
meta = pd.read_csv(OUT + r"\split.csv")
emb = np.load(OUT + r"\emb_resnet18.npy")   # same row order as split.csv

# here i turn "unknown" into real missing values
meta["sex"] = meta["sex"].replace("unknown", np.nan)
meta["localization"] = meta["localization"].replace("unknown", np.nan)

# here i put the patient details and the 512 image features in one table
emb_cols = [f"e{i}" for i in range(emb.shape[1])]
emb_df = pd.DataFrame(emb, columns=emb_cols)
data = pd.concat([meta[["age", "sex", "localization"]], emb_df], axis=1)

# here i split by the split column, test is not used in this script
train_mask = (meta.split == "train").values
val_mask = (meta.split == "val").values
X_train, y_train = data[train_mask], meta.dx[train_mask]
X_val, y_val = data[val_mask], meta.dx[val_mask]

def build(kind, C):
    # here i build the pipeline for one kind of input: "meta", "image" or "both"
    # every step is fitted on train data only when i call fit()
    parts = []
    if kind in ["meta", "both"]:
        parts.append(("age", Pipeline([
            ("fill", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())]), ["age"]))
        parts.append(("cat", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["sex", "localization"]))
    if kind in ["image", "both"]:
        parts.append(("img", StandardScaler(), emb_cols))
    return Pipeline([
        ("prep", ColumnTransformer(parts)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=C)),
    ])

names = {"meta": "metadata only (LR)", "image": "image only (ResNet18+LR)",
         "both": "hybrid concat (ResNet18+meta+LR)"}

for kind in ["meta", "image", "both"]:
    print("\n=====", names[kind], "=====")
    best_f1, best_model, best_C = -1, None, None

    # here i try 4 values of C (smaller C = stronger regularization)
    # i compare train F1 and val F1 to see overfitting
    for C in [0.001, 0.01, 0.1, 1]:
        model = build(kind, C)
        model.fit(X_train, y_train)
        f1_train = f1_score(y_train, model.predict(X_train), average="macro")
        f1_val = f1_score(y_val, model.predict(X_val), average="macro")
        print("C =", C, "| train macro F1:", round(f1_train, 3), "| val macro F1:", round(f1_val, 3))
        if f1_val > best_f1:
            best_f1, best_model, best_C = f1_val, model, C

    # here i keep the C that gave the best validation F1 and save its result
    print("best C (chosen on validation):", best_C)
    result = evaluate(names[kind], y_val, best_model.predict(X_val))
    result["best_C"] = best_C
    save_result(result)