import pandas as pd, numpy as np
from utils import to_lesion_level

OUT = r"D:\ML CEP\CEP_CODE"
meta = pd.read_csv(OUT + r"\split.csv")
val_mask = (meta.split == "val").values
classes = sorted(meta.dx.unique())

table = {}
for loss in ["plain", "weighted", "focal"]:
    per_seed = []
    for seed in [42, 1, 2]:
        proba = np.load(OUT + r"\outputs\effb0_{}_aug1_seed{}_val_proba.npy".format(loss, seed))
        mp, pred, labels = to_lesion_level(proba, meta.lesion_id[val_mask], meta.dx[val_mask], classes)
        # here i calculate the recall of every class: share of its lesions that got the right prediction
        per_seed.append([(pred[labels == c] == c).mean() for c in classes])
    # here i take the mean over the 3 seeds
    table[loss] = np.mean(per_seed, axis=0)

out = pd.DataFrame(table, index=classes).round(3)
out["n_lesions"] = [(labels == c).sum() for c in classes]
print(out)