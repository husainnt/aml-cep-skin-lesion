import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(r"D:\ML CEP\dataverse_files")
meta = pd.read_csv(ROOT / "HAM10000_metadata")   # file has no extension

img_paths = list(ROOT.glob("HAM10000_images_part_*/*.jpg"))
img_ids = {p.stem for p in img_paths}

print("metadata rows:", len(meta), "| image files:", len(img_ids))
print("meta ids without image:", len(set(meta.image_id) - img_ids))
print("images without meta:", len(img_ids - set(meta.image_id)))
print("duplicate image_ids:", meta.image_id.duplicated().sum())
print("unique lesions:", meta.lesion_id.nunique())
print("\nimages per lesion:\n", meta.groupby("lesion_id").size().value_counts().sort_index())
print("\nclass counts:\n", meta.dx.value_counts())
print("\nmissing values:\n", meta.isna().sum())
print("\nsex:\n", meta.sex.value_counts())
print("\nlocalization:\n", meta.localization.value_counts())
print("\ndx_type (how labels were confirmed):\n", meta.dx_type.value_counts())