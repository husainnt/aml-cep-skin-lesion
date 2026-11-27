import pandas as pd, numpy as np, os, json
import matplotlib.pyplot as plt
from utils import to_lesion_level

OUT = r"D:\ML CEP\CEP_CODE"
HIGH = ["mel", "bcc", "akiec"]

# ---------- my two decisions (i change them after i have seen the tables, and write the reason in the log) ----------
CAP = 0.25            # largest share of all lesions that the doctors can see (referred + deferred)
DEFER_SHARE = 0.10    # share of lesions where the system says "i am not sure" and defers
# ------------------------------------------------------------------------------------------------------------------

rng = np.random.default_rng(42)
os.makedirs(OUT + r"\figures", exist_ok=True)

meta = pd.read_csv(OUT + r"\split.csv")
val_mask = (meta.split == "val").values
val_lesions = meta.lesion_id[val_mask]
val_names = meta.dx[val_mask]
classes = sorted(meta.dx.unique())
high_idx = [classes.index(c) for c in HIGH]

# here i list all 6 models (2 designs x 3 seeds), the 2 deployed ones are used for the plots and the app
files = {}
for seed in [42, 1, 2]:
    files["image only seed%d" % seed] = r"outputs\effb0_plain_aug1_seed%d_val_proba.npy" % seed
    files["dropout hybrid seed%d" % seed] = r"outputs\hyb_effb0_plain_md0.3_seed%d_val_proba.npy" % seed
deployed = ["image only seed1", "dropout hybrid seed42"]

# here i make one probability vector per lesion (average of its images)
data = {}
for name, path in files.items():
    proba = np.load(os.path.join(OUT, path))
    mp, pred, labels = to_lesion_level(proba, val_lesions, val_names, classes)
    data[name] = mp.values
y_true = labels.values
y_idx = np.array([classes.index(c) for c in y_true])
is_high = labels.isin(HIGH).values
n = len(y_true)
print("validation lesions:", n, "| high concern:", is_high.sum())

# ===================== 1. calibration =====================
def ece(probs, y_idx, n_bins=10):
    # here i split the confidences into bins and average |accuracy - confidence| (weighted by bin size)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_idx).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        in_bin = (conf > bins[i]) & (conf <= bins[i + 1])
        if in_bin.sum() > 0:
            total += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return total

def reliability_points(probs, y_idx, n_bins=10):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_idx).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        in_bin = (conf > bins[i]) & (conf <= bins[i + 1])
        if in_bin.sum() > 0:
            xs.append(conf[in_bin].mean())
            ys.append(correct[in_bin].mean())
    return xs, ys

def apply_T(probs, T):
    # here i use log(probability) as the logits and divide them by the temperature T
    logits = np.log(np.clip(probs, 1e-9, 1)) / T
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)

def nll(probs, y_idx):
    return -np.mean(np.log(np.clip(probs[np.arange(len(y_idx)), y_idx], 1e-9, 1)))

def fit_T(probs, y_idx):
    # here i try many values of T and keep the one with the lowest negative log likelihood
    grid = np.arange(0.5, 5.01, 0.01)
    losses = [nll(apply_T(probs, T), y_idx) for T in grid]
    return grid[int(np.argmin(losses))]

cal_all, rows = {}, []
for name, probs in list(data.items()):
    T = fit_T(probs, y_idx)
    cal_all[name] = apply_T(probs, T)
    # here i also do a cross-fitted check: T is fitted on 4/5 of the lesions and applied to the other 1/5
    folds = np.zeros(n, dtype=int)
    folds[rng.permutation(n)] = np.arange(n) % 5
    cal_cv = np.zeros_like(probs)
    for f in range(5):
        te = folds == f
        cal_cv[te] = apply_T(probs[te], fit_T(probs[~te], y_idx[~te]))
    rows.append({"model": name, "T": round(T, 2), "ECE before": round(ece(probs, y_idx), 3),
                 "ECE after (T fitted on all val)": round(ece(cal_all[name], y_idx), 3),
                 "ECE after (cross-fitted)": round(ece(cal_cv, y_idx), 3),
                 "NLL before": round(nll(probs, y_idx), 3),
                 "NLL after": round(nll(cal_all[name], y_idx), 3)})
    data[name + "_cv"] = cal_cv
print("\n=== calibration (validation lesions) ===")
print(pd.DataFrame(rows).to_string(index=False))

# here i draw the reliability diagrams of the two deployed models (before and after, cross-fitted)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, name in zip(axes, deployed):
    x0, y0 = reliability_points(data[name], y_idx)
    x1, y1 = reliability_points(data[name + "_cv"], y_idx)
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.plot(x0, y0, "o-", label="before")
    ax.plot(x1, y1, "s-", label="after temperature scaling")
    ax.set_title(name)
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.legend(); ax.grid(alpha=0.3)
plt.savefig(OUT + r"\figures\fig_reliability.png", dpi=200, bbox_inches="tight")

# ===================== 2. referral threshold and deferral =====================
def find_t(p_high, deferred, cap):
    # here i start with a very high threshold and lower it while the total referral rate stays under the cap
    best_t = 1.01
    for t in np.sort(np.unique(p_high))[::-1]:
        referred = deferred | (p_high >= t)
        if referred.mean() <= cap:
            best_t = t
        else:
            break
    return best_t

def evaluate_rule(cal, tau, t):
    p_high = cal[:, high_idx].sum(axis=1)
    conf = cal.max(axis=1)
    pred = cal.argmax(axis=1)
    deferred = conf < tau                       # "i am not sure" -> doctor
    referred = deferred | (p_high >= t)         # deferred lesions also count as referred
    kept = ~deferred
    return {"deferred share": round(deferred.mean(), 3),
            "referral rate": round(referred.mean(), 3),
            "ref sensitivity": round(referred[is_high].mean(), 3),
            "missed high-concern": int((~referred & is_high).sum()),
            "error among kept": round((pred[kept] != y_idx[kept]).mean(), 3),
            "errors in deferred": int((pred[deferred] != y_idx[deferred]).sum())}

print("\n=== deferral table, cap = %.2f (deployed models) ===" % CAP)
for name in deployed:
    cal = cal_all[name]
    p_high = cal[:, high_idx].sum(axis=1)
    conf = cal.max(axis=1)
    table = []
    for share in [0.0, 0.05, 0.10, 0.15, 0.20]:
        tau = np.quantile(conf, share) if share > 0 else 0.0
        t = find_t(p_high, conf < tau, CAP)
        row = {"target defer share": share, "tau": round(tau, 3), "t": round(t, 3)}
        row.update(evaluate_rule(cal, tau, t))
        table.append(row)
    print("\n" + name)
    print(pd.DataFrame(table).to_string(index=False))

# ===================== 3. my chosen rule for all 6 models (saved as a DRAFT) =====================
rules = {}
for name in files:
    cal = cal_all[name]
    p_high = cal[:, high_idx].sum(axis=1)
    conf = cal.max(axis=1)
    tau = np.quantile(conf, DEFER_SHARE) if DEFER_SHARE > 0 else 0.0
    t = find_t(p_high, conf < tau, CAP)
    T = fit_T(data[name], y_idx)
    rules[name] = {"T": float(T), "tau": float(tau), "t": float(t), "cap": CAP, "defer_share": DEFER_SHARE}
    print(name, "| rule on validation:", evaluate_rule(cal, tau, t))
with open(OUT + r"\decision_rule_DRAFT.json", "w") as f:
    json.dump(rules, f, indent=2)
print("\nsaved decision_rule_DRAFT.json (rename it only when i freeze the rules)")

# ===================== 4. referral curve and risk-coverage curve =====================
plt.figure(figsize=(6, 5))
for name in deployed:
    p_high = cal_all[name][:, high_idx].sum(axis=1)
    order = np.argsort(-p_high)
    plt.plot(np.arange(1, n + 1) / n, np.cumsum(is_high[order]) / is_high.sum(), label=name)
plt.plot([0, 1], [0, 1], "k--", label="random referral")
plt.axvline(CAP, color="gray", linestyle=":", label="my cap")
plt.xlabel("referral rate"); plt.ylabel("referral sensitivity")
plt.title("Referral curve (validation, calibrated)")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(OUT + r"\figures\fig_referral_curve_final.png", dpi=200, bbox_inches="tight")

plt.figure(figsize=(6, 5))
print("\n=== risk-coverage (error rate among the kept lesions) ===")
for name in deployed:
    cal = cal_all[name]
    order = np.argsort(-cal.max(axis=1))                     # most confident first
    wrong = (cal.argmax(axis=1) != y_idx)[order]
    risk = np.cumsum(wrong) / np.arange(1, n + 1)
    coverage = np.arange(1, n + 1) / n
    plt.plot(coverage, risk, label=name)
    for c in [1.0, 0.9, 0.8]:
        print(name, "| coverage %.2f -> error %.3f" % (c, risk[int(c * n) - 1]))
plt.xlabel("coverage (share of lesions kept)"); plt.ylabel("error rate among kept")
plt.title("Risk-coverage curve (validation, calibrated)")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(OUT + r"\figures\fig_risk_coverage.png", dpi=200, bbox_inches="tight")

# ===================== 5. cost for several values of r =====================
print("\n=== cost = r x (missed high-concern) + (needless referrals), per 100 lesions ===")
for r in [2, 5, 10, 20]:
    for name in deployed:
        p_high = cal_all[name][:, high_idx].sum(axis=1)
        order = np.argsort(-p_high)
        cum_high = np.cumsum(is_high[order])        # high-concern lesions inside the top k
        k = np.arange(1, n + 1)
        cost = r * (is_high.sum() - cum_high) + (k - cum_high)
        k_best = int(np.argmin(cost))
        k_cap = int(CAP * n) - 1
        print("r=%2d %-24s | best rate %.3f (sens %.3f, cost %.1f) | at cap %.2f (sens %.3f, cost %.1f)" %
              (r, name, (k_best + 1) / n, cum_high[k_best] / is_high.sum(), 100 * cost[k_best] / n,
               CAP, cum_high[k_cap] / is_high.sum(), 100 * cost[k_cap] / n))