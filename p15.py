import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import time, json, joblib, platform, os
from pathlib import Path
from PIL import Image
from torchvision import models

ROOT = Path(r"D:\ML CEP\dataverse_files")
OUT = Path(r"D:\ML CEP\CEP_CODE")
HIGH = ["mel", "bcc", "akiec"]
N_IMAGES, WARMUP = 60, 5
rng = np.random.default_rng(0)

print("cpu:", platform.processor(), "| torch threads:", torch.get_num_threads(), "| torch", torch.__version__)

meta = pd.read_csv(OUT / "split.csv")
classes = sorted(meta.dx.unique())
high_idx = [classes.index(c) for c in HIGH]
paths = {p.stem: p for p in ROOT.glob("HAM10000_images_part_*/*.jpg")}

# here i take a random sample of validation images
sample = meta[meta.split == "val"].sample(N_IMAGES + WARMUP, random_state=0).to_dict("records")

# here i load the rules (frozen if it exists, else the draft)
rule_file = OUT / "decision_rule_FROZEN.json"
if not rule_file.exists():
    rule_file = OUT / "decision_rule_DRAFT.json"
print("using rules from:", rule_file.name)
rules = json.load(open(rule_file))

prep = joblib.load(OUT / "outputs" / "meta_prep.joblib")

class Hybrid(nn.Module):
    def __init__(self, n_meta):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        base.classifier = nn.Identity()
        self.image_branch = base
        self.meta_branch = nn.Sequential(nn.Linear(n_meta, 32), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280 + 32, 7))
    def forward(self, img, m):
        return self.head(torch.cat([self.image_branch(img), self.meta_branch(m)], dim=1))

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def load_image(image_id):
    # here i read the jpg, resize it to 224x224 and normalize it
    img = Image.open(paths[image_id]).convert("RGB").resize((224, 224), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0) / 255
    return (x - mean) / std

def meta_features(r):
    # here i make the patient details of one lesion, with the same cleaning as in training
    sex = np.nan if r["sex"] == "unknown" else r["sex"]
    loc = np.nan if r["localization"] == "unknown" else r["localization"]
    d = pd.DataFrame({"age": [r["age"]], "sex": pd.Series([sex], dtype=object),
                      "localization": pd.Series([loc], dtype=object)})
    d["age_missing"] = int(pd.isna(r["age"]))
    d["sex_missing"] = int(pd.isna(sex))
    d["loc_missing"] = int(pd.isna(loc))
    return torch.from_numpy(prep.transform(d).astype(np.float32))

def decide(probs, rule):
    # here i apply the temperature and the two thresholds of the frozen rule
    logits = np.log(np.clip(probs, 1e-9, 1)) / rule["T"]
    e = np.exp(logits - logits.max())
    cal = e / e.sum()
    if cal.max() < rule["tau"]:
        return "deferred"
    return "referred" if cal[high_idx].sum() >= rule["t"] else "not referred"

def load_model(design, seed):
    if design == "image only":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(1280, 7)
        f = OUT / "outputs" / f"effb0_plain_aug1_seed{seed}.pt"
    else:
        m = Hybrid(19)
        f = OUT / "outputs" / f"hyb_effb0_plain_md0.3_seed{seed}.pt"
    m.load_state_dict(torch.load(f, map_location="cpu"))
    return m.eval(), f

for design, seed in [("image only", 1), ("dropout hybrid", 42)]:
    model, f = load_model(design, seed)
    rule = rules[f"{design} seed{seed}"]
    t_read, t_meta, t_net, t_dec, t_all, decisions = [], [], [], [], [], []
    for k, r in enumerate(sample):
        a = time.perf_counter()
        x = load_image(r["image_id"])
        b = time.perf_counter()
        m = meta_features(r) if design == "dropout hybrid" else None
        c = time.perf_counter()
        with torch.no_grad():
            logits = model(x, m) if m is not None else model(x)
            probs = F.softmax(logits, dim=1)[0].numpy()
        d = time.perf_counter()
        dec = decide(probs, rule)
        e = time.perf_counter()
        if k >= WARMUP:        # the first runs are warm-up and are not counted
            t_read.append(b - a); t_meta.append(c - b); t_net.append(d - c)
            t_dec.append(e - d); t_all.append(e - a); decisions.append(dec)
    ms = lambda v: np.array(v) * 1000
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{design} seed {seed}: parameters {n_params:,} | file {os.path.getsize(f) / 1e6:.1f} MB")
    print("  total per lesion (ms): mean %.0f | median %.0f | p95 %.0f | max %.0f" %
          (ms(t_all).mean(), np.median(ms(t_all)), np.percentile(ms(t_all), 95), ms(t_all).max()))
    print("  mean per stage (ms): read+resize %.0f | metadata %.1f | network %.0f | decision %.2f" %
          (ms(t_read).mean(), ms(t_meta).mean(), ms(t_net).mean(), ms(t_dec).mean()))
    print("  decisions on the sample:", pd.Series(decisions).value_counts().to_dict())
    print("  R5 check: p95 under 1000 ms?", np.percentile(ms(t_all), 95) < 1000,
          "| file under 100 MB?", os.path.getsize(f) / 1e6 < 100)