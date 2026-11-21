import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

# here i load the split file i saved before
meta = pd.read_csv(r"D:\ML CEP\CEP_CODE\split.csv")

# here i turn the "unknown" text into real missing values (NaN) so the imputer can fill them
meta["sex"] = meta["sex"].replace("unknown", np.nan)
meta["localization"] = meta["localization"].replace("unknown", np.nan)

# here i take only train and val, i do NOT touch the test set
train = meta[meta.split == "train"]
val = meta[meta.split == "val"]

# here i choose the patient details as input (X) and the diagnosis as output (y)
features = ["age", "sex", "localization"]
X_train, y_train = train[features], train["dx"]
X_val, y_val = val[features], val["dx"]

# here i build the preprocessing for the number column (age):
# fill missing with the median, then scale. both are learned from train only
num_part = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

# here i build the preprocessing for the text columns (sex, localization):
# fill missing with the most common value, then one-hot encode
# handle_unknown="ignore" means a category not seen in training will not crash it
cat_part = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

prep = ColumnTransformer([
    ("num", num_part, ["age"]),
    ("cat", cat_part, ["sex", "localization"]),
])

# here i put preprocessing and the model in one pipeline
# class_weight="balanced" makes the rare classes count more during training
model = Pipeline([
    ("prep", prep),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

# here i train. fit() learns the preprocessing and the model from train data only
model.fit(X_train, y_train)

# here i make a silly model that always predicts the most common class (nv)
# it shows why accuracy alone is a bad measure
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)

high_concern = ["mel", "bcc", "akiec"]

def report(name, pred, y_true):
    # here i calculate the normal measures
    print("\n==", name, "==")
    print("accuracy:", round(accuracy_score(y_true, pred), 3))
    print("macro F1:", round(f1_score(y_true, pred, average="macro"), 3))
    # here i calculate the two referral measures from the CEP
    # a lesion is referred if the model predicts a high concern class
    referred = pd.Series(pred).isin(high_concern).values
    is_high = y_true.isin(high_concern).values
    print("referral rate:", round(referred.mean(), 3))
    print("referral sensitivity:", round(referred[is_high].mean(), 3))

report("always predict nv", dummy.predict(X_val), y_val)
report("metadata only (logistic regression)", model.predict(X_val), y_val)

# here i print precision / recall for every class
print(classification_report(y_val, model.predict(X_val), zero_division=0))