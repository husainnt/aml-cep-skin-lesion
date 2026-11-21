import pandas as pd, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

OUT = Path(r"D:\ML CEP\CEP_CODE")
meta = pd.read_csv(OUT / "split.csv")
features = np.load(OUT / "emb_resnet18.npy")   # same row order as split.csv

# here i pick the rows of train and val using the split column (test stays untouched)
train_mask = (meta.split == "train").values
val_mask = (meta.split == "val").values
X_train, y_train = features[train_mask], meta.dx[train_mask]
X_val, y_val = features[val_mask], meta.dx[val_mask]

# here i scale the features (learned from train only) and train logistic regression
model = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
])
model.fit(X_train, y_train)
pred = model.predict(X_val)

high_concern = ["mel", "bcc", "akiec"]
# here i calculate the same measures as before so the baselines can be compared
referred = pd.Series(pred).isin(high_concern).values
is_high = y_val.isin(high_concern).values
print("accuracy:", round(accuracy_score(y_val, pred), 3))
print("macro F1:", round(f1_score(y_val, pred, average="macro"), 3))
print("referral rate:", round(referred.mean(), 3))
print("referral sensitivity:", round(referred[is_high].mean(), 3))
print(classification_report(y_val, pred, zero_division=0))