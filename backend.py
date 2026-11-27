"""Backend of the skin-lesion triage app (educational prototype, not a medical device).

predict(tab, image=None, age=None, sex=None, location=None) -> dict
  tab      : 'metadata', 'image' or 'hybrid'
  image    : PIL image or None
  age      : number or None, sex: 'male' / 'female' / None, location: one of LOCATIONS or None
It never raises: on any problem it returns {'ok': False, 'error': '...'}.
time_ms is the time of the prediction (preprocessing + network + decision), heatmap_ms is extra.
"""
import os, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, joblib
from PIL import Image
from torchvision import models

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HIGH = ["mel", "bcc", "akiec"]
HIGH_IDX = [CLASSES.index(c) for c in HIGH]
LOCATIONS = ["abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital", "hand",
             "lower extremity", "neck", "scalp", "trunk", "upper extremity"]
MIN_SIDE = 64                      # smaller images are refused

HERE = Path(__file__).resolve().parent
SEARCH = [Path(os.environ["TRIAGE_MODEL_DIR"])] if os.environ.get("TRIAGE_MODEL_DIR") else []
SEARCH += [HERE, HERE / "models", HERE / "outputs", HERE.parent, HERE.parent / "models", HERE.parent / "outputs"]

def _find(name):
    # here i look for a file in the usual folders, so the app works from the repo or from a copy
    for d in SEARCH:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError("file not found: " + name)

class Hybrid(nn.Module):
    # same model as in train_hybrid.py
    def __init__(self, n_meta):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        base.classifier = nn.Identity()
        self.image_branch = base
        self.meta_branch = nn.Sequential(nn.Linear(n_meta, 32), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280 + 32, 7))
    def forward(self, img, m):
        return self.head(torch.cat([self.image_branch(img), self.meta_branch(m)], dim=1))

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_STATE = {}

def warmup():
    """Load everything once (the front end can call this at start-up)."""
    return _load()

def _load():
    if _STATE:
        return _STATE
    # the two FINAL LOCKED models and the frozen rules
    rules = json.load(open(_find("decision_rule_FROZEN.json")))
    img = models.efficientnet_b0(weights=None)
    img.classifier[1] = nn.Linear(1280, 7)
    img.load_state_dict(torch.load(_find("effb0_plain_aug1_seed1.pt"), map_location="cpu"))
    hyb = Hybrid(19)
    hyb.load_state_dict(torch.load(_find("hyb_effb0_plain_md0.3_seed42.pt"), map_location="cpu"))
    _STATE.update({"rules": rules, "img": img.eval(), "hyb": hyb.eval(),
                   "img_rule": rules["image only seed1"], "hyb_rule": rules["dropout hybrid seed42"],
                   "prep": joblib.load(_find("meta_prep.joblib")), "comp": None, "gate": None, "resnet": None})
    try:                                   # comparison models are optional
        _STATE["comp"] = joblib.load(_find("comparison_models.joblib"))
        _STATE["comp_rules"] = json.load(open(_find("decision_rule_comparison.json")))
    except FileNotFoundError:
        pass
    try:                                   # image-quality gate is optional
        _STATE["gate"] = json.load(open(_find("quality_gate.json")))
    except FileNotFoundError:
        pass
    return _STATE

def _resnet():
    if _STATE["resnet"] is None:
        r = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        r.fc = nn.Identity()
        _STATE["resnet"] = r.eval()
    return _STATE["resnet"]

# ---------------- small helpers ----------------
def _apply_T(p, T):
    z = np.log(np.clip(p, 1e-9, 1)) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def _decide(cal, rule):
    # here i apply the thresholds of the rule: deferred if not confident, referred if p_high is large enough
    if cal.max() < rule["tau"]:
        return "deferred"
    return "referred" if cal[HIGH_IDX].sum() >= rule["t"] else "not referred"

def _result(model, cal, time_ms, decision, heatmap=None, reason="model rule"):
    return {"model": model, "probs": {c: float(cal[i]) for i, c in enumerate(CLASSES)},
            "pred": CLASSES[int(cal.argmax())], "confidence": float(cal.max()),
            "decision": decision, "decision_reason": reason, "p_high": float(cal[HIGH_IDX].sum()),
            "time_ms": float(time_ms), "heatmap": heatmap}

def _check_inputs(tab, image, age, sex, location):
    if tab not in ("metadata", "image", "hybrid"):
        raise ValueError("Unknown tab: %r" % (tab,))
    if age is not None:
        try:
            age = float(age)
        except (TypeError, ValueError):
            raise ValueError("Age must be a number.")
        if np.isnan(age):
            age = None
        elif not (0 <= age <= 120):
            raise ValueError("Age must be between 0 and 120.")
    sex = None if sex is None else str(sex).strip().lower()
    if sex in ("", "not given", "unknown", "none"):
        sex = None
    if sex not in (None, "male", "female"):
        raise ValueError("Sex must be male, female or left empty.")
    location = None if location is None else str(location).strip().lower()
    if location in ("", "not given", "unknown", "none"):
        location = None
    if location not in [None] + LOCATIONS:
        raise ValueError("Unknown body location: %s" % location)
    if tab in ("image", "hybrid"):
        if image is None:
            raise ValueError("Please upload an image.")
        if not isinstance(image, Image.Image):
            raise ValueError("The uploaded file is not an image.")
    return age, sex, location

def _prep_image(image):
    try:
        im = image.convert("RGB")
    except Exception:
        raise ValueError("The image could not be read (the file may be corrupt).")
    if min(im.size) < MIN_SIDE:
        raise ValueError("The image is too small (at least %d x %d pixels)." % (MIN_SIDE, MIN_SIDE))
    arr = np.array(im.resize((224, 224), Image.BILINEAR))             # same resize as in training
    x = (torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255 - MEAN) / STD
    return arr, x

def _meta_frame(age, sex, loc):
    return pd.DataFrame({"age": [np.nan if age is None else age],
                         "sex": pd.Series([np.nan if sex is None else sex], dtype=object),
                         "localization": pd.Series([np.nan if loc is None else loc], dtype=object)})

def _meta_tensor(age, sex, loc):
    # here i make the 19 patient-detail features of the hybrid, like in training (with the missing flags)
    d = _meta_frame(age, sex, loc)
    d["age_missing"] = int(age is None)
    d["sex_missing"] = int(sex is None)
    d["loc_missing"] = int(loc is None)
    return torch.from_numpy(_STATE["prep"].transform(d).astype(np.float32))

def _quality(arr):
    # here i check sharpness and brightness with thresholds chosen on validation images
    gate = _STATE["gate"]
    if gate is None:
        return []
    g = arr.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    msgs = []
    if lap.var() < gate["sharp_min"]:
        msgs.append("The image looks blurry. Please retake the photo in focus.")
    b = arr.mean()
    if b < gate["bright_min"]:
        msgs.append("The image is very dark. Please retake the photo with more light.")
    if b > gate["bright_max"]:
        msgs.append("The image is very bright or washed out. Please retake the photo.")
    return msgs

def _gradcam(model, x, m, hybrid, cls):
    # here i make a Grad-CAM heatmap of the last convolution block for the predicted class
    layer = model.image_branch.features[-1] if hybrid else model.features[-1]
    store = {}
    def fwd(mod, inp, out):
        store["a"] = out.detach()
        out.register_hook(lambda g: store.__setitem__("g", g.detach()))
    handle = layer.register_forward_hook(fwd)
    try:
        model.zero_grad()
        logits = model(x, m) if hybrid else model(x)
        logits[0, cls].backward()
    finally:
        handle.remove()
    a, g = store["a"][0], store["g"][0]
    cam = F.relu((g.mean(dim=(1, 2))[:, None, None] * a).sum(0))
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
    return cam

def _overlay(arr, cam):
    # here i colour the heatmap (jet-like) and blend it with the 224 x 224 image
    c = cam
    heat = np.stack([np.clip(1.5 - np.abs(4 * c - 3), 0, 1),
                     np.clip(1.5 - np.abs(4 * c - 2), 0, 1),
                     np.clip(1.5 - np.abs(4 * c - 1), 0, 1)], axis=-1)
    mix = 0.6 * arr.astype(np.float32) + 0.4 * heat * 255
    return Image.fromarray(np.clip(mix, 0, 255).astype(np.uint8))

def _run_cnn(model, x, m, hybrid):
    with torch.no_grad():
        logits = model(x, m) if hybrid else model(x)
    return F.softmax(logits, dim=1)[0].numpy()

def _run_comparison(name, tab, emb, emb_ms, df):
    rec, rule = _STATE["comp"][name], _STATE["comp_rules"][name]
    t0 = time.perf_counter()
    Xm = rec["meta_prep"].transform(df)
    X = {"meta": Xm, "emb": emb[None, :] if emb is not None else None,
         "emb+meta": np.hstack([emb[None, :], Xm]) if emb is not None else None}[rec["kind"]]
    cal = _apply_T(rec["clf"].predict_proba(X)[0], rule["T"])
    ms = (time.perf_counter() - t0) * 1000 + (emb_ms if rec["kind"] != "meta" else 0.0)
    return _result(name, cal, ms, _decide(cal, rule), reason="comparison rule, not locked")

# ---------------- the one function of the contract ----------------
def predict(tab, image=None, age=None, sex=None, location=None):
    try:
        return _predict(tab, image, age, sex, location)
    except Exception as e:                               # never crash the app
        return {"ok": False, "error": str(e) if isinstance(e, ValueError) else
                "Something went wrong while predicting (%s)." % e, "warnings": [], "final": None, "others": []}

def _predict(tab, image, age, sex, loc):
    age, sex, loc = _check_inputs(tab, image, age, sex, loc)
    S = _load()
    warnings, others = [], []
    df = _meta_frame(age, sex, loc)
    comp_names = {"metadata": ["metadata random forest", "metadata gradient boosting"],
                  "image": ["ResNet18 + logistic regression (image)"],
                  "hybrid": ["ResNet18 + logistic regression (image + details)"]}[tab]

    if tab == "metadata":
        if age is None and sex is None and loc is None:
            warnings.append("No details were entered, so this is only the overall class frequency.")
        if S["comp"] is None:
            raise ValueError("The metadata model file (comparison_models.joblib) is missing.")
        final = _run_comparison("metadata logistic regression", tab, None, 0.0, df)
        final["model"] = "Metadata-only logistic regression (baseline, weak)"
        final["decision_reason"] = "metadata rule fitted on validation"
        arr = None
    else:
        t0 = time.perf_counter()
        arr, x = _prep_image(image)
        hybrid = tab == "hybrid"
        m = _meta_tensor(age, sex, loc) if hybrid else None
        model, rule = (S["hyb"], S["hyb_rule"]) if hybrid else (S["img"], S["img_rule"])
        cal = _apply_T(_run_cnn(model, x, m, hybrid), rule["T"])
        decision = _decide(cal, rule)
        ms = (time.perf_counter() - t0) * 1000
        reason = "model rule"
        bad = _quality(arr)
        if bad:
            warnings += bad
            if decision != "deferred":
                decision, reason = "deferred", "image quality gate"
        if hybrid and age is None and sex is None and loc is None:
            warnings.append("No patient details were given, so the hybrid model relies on the image alone.")
        final = _result("Dropout hybrid EfficientNet-B0 (final)" if hybrid else "EfficientNet-B0, image only (final)",
                        cal, ms, decision, reason=reason)
        h0 = time.perf_counter()
        try:                                              # a heatmap problem must not hide the result
            cam = _gradcam(model, x, m, hybrid, int(cal.argmax()))
            final["heatmap"] = _overlay(arr, cam)
        except Exception as e:
            warnings.append("The heatmap could not be made (%s)." % e)
        final["heatmap_ms"] = (time.perf_counter() - h0) * 1000
        if hybrid:                                        # the other deployed model, comparison only
            t1 = time.perf_counter()
            c2 = _apply_T(_run_cnn(S["img"], x, None, False), S["img_rule"]["T"])
            others.append(_result("EfficientNet-B0, image only (deployed, comparison)", c2,
                                  (time.perf_counter() - t1) * 1000, _decide(c2, S["img_rule"]),
                                  reason="frozen rule"))

    # comparison models (optional, never allowed to break the main result)
    if S["comp"] is not None:
        try:
            emb, emb_ms = None, 0.0
            if tab != "metadata":
                t2 = time.perf_counter()
                with torch.no_grad():
                    emb = _resnet()(_prep_image(image)[1])[0].numpy()
                emb_ms = (time.perf_counter() - t2) * 1000
            for name in comp_names:
                others.append(_run_comparison(name, tab, emb, emb_ms, df))
        except Exception as e:
            warnings.append("Comparison models are not available (%s)." % e)
    return {"ok": True, "error": None, "warnings": warnings, "final": final, "others": others}
