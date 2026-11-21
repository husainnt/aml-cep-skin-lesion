import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
ROOT = Path(r"D:\ML CEP\dataverse_files")
meta = pd.read_csv(ROOT / "HAM10000_metadata")
SEED=42 # here is fix the seed so that split is same everytime
# here i check that every lesion has one diagnosis
# if a lesion had two different labels, grouping by lesion would be confusing
dx_per_lesion=meta.groupby("lesion_id")["dx"].nunique()
print("lesions with more than one dx:", (dx_per_lesion > 1).sum())

# here i count classes at lesion level (one row per lesion)
lesions = meta.drop_duplicates("lesion_id")
print("\nclass counts at lesion level:\n", lesions.dx.value_counts())

# here i look at the dataset column, it may be a hidden shortcut (different sources)
print("\ndataset column:\n", meta.dataset.value_counts())
print("\ndx by dataset:\n", pd.crosstab(meta.dataset, meta.dx))

# here i split into 7 folds, grouped by lesion_id and stratified by dx
# fold 0 will be test, fold 1 validation, the other 5 folds train (about 71/14/14)
sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
meta["split"] = "train"
for fold, (_, idx) in enumerate(sgkf.split(meta, meta["dx"], groups=meta["lesion_id"])):
    if fold == 0:
        meta.loc[meta.index[idx], "split"] = "test"
    elif fold == 1:
        meta.loc[meta.index[idx], "split"] = "val"

# here i print how big each part is and the class counts in each
print("\nsplit sizes:\n", meta.split.value_counts())
print("\nclass counts per split:\n", pd.crosstab(meta.split, meta.dx))

# here i PROVE there is no leakage: no lesion_id appears in more than one split
lesion_sets = {s: set(meta[meta.split == s].lesion_id) for s in ["train", "val", "test"]}
assert lesion_sets["train"].isdisjoint(lesion_sets["val"])
assert lesion_sets["train"].isdisjoint(lesion_sets["test"])
assert lesion_sets["val"].isdisjoint(lesion_sets["test"])
# here i also check every class appears in every split
for s in ["train", "val", "test"]:
    assert meta[meta.split == s].dx.nunique() == 7
print("\nno lesion shared between splits, all 7 classes in every split")

# here i save the split so every later script uses exactly this file
meta.to_csv(r"D:\ML CEP\CEP_CODE\split.csv", index=False)