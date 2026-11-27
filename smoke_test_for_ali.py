# smoke_test_for_ali.py
# a quick check that backend.py works on this PC and follows the predict() contract
# it needs only the files in app_code.zip and models.zip (no dataset)
# run it from the folder that holds backend.py:  python -u smoke_test_for_ali.py

import sys, traceback
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backend import predict

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
fails = []


def check(label, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + label + (("  (" + str(detail) + ")") if detail else ""))
    if not ok:
        fails.append(label)


def check_contract(res, name, want_heatmap, want_others):
    # here i check the dictionary has everything the front end needs
    check(name + ": result is a dict", isinstance(res, dict))
    if not isinstance(res, dict):
        return
    check(name + ": ok is True", res.get("ok") is True, res.get("error"))
    check(name + ": warnings is a list", isinstance(res.get("warnings"), list))
    fin = res.get("final") or {}
    for k in ["model", "probs", "pred", "confidence", "decision", "p_high", "time_ms", "decision_reason"]:
        check(name + ": final has " + k, k in fin)
    probs = fin.get("probs") or {}
    check(name + ": 7 class probabilities", set(probs) == set(CLASSES), sorted(probs))
    if probs:
        check(name + ": probabilities sum to 1", abs(sum(probs.values()) - 1) < 1e-3, round(sum(probs.values()), 4))
    check(name + ": decision is allowed", fin.get("decision") in ("referred", "deferred", "not referred"), fin.get("decision"))
    if want_heatmap:
        check(name + ": heatmap present", fin.get("heatmap") is not None)
    others = res.get("others")
    check(name + ": others is a list", isinstance(others, list))
    # my backend test showed 2 / 1 / 2 comparison rows on the metadata / image / hybrid tabs
    check(name + ": comparison rows", isinstance(others, list) and len(others) == want_others,
          f"{len(others or [])} rows, expected {want_others}")
    print(f"      -> {fin.get('pred')} conf {fin.get('confidence')} decision {fin.get('decision')} "
          f"time {fin.get('time_ms')} ms, warnings {res.get('warnings')}")


demo_dir = HERE / "demo_images"
demos = sorted(p for p in demo_dir.glob("*.jpg") if "blurred" not in p.name)
blurred = demo_dir / "demo_blurred_radius2.jpg"
check("demo images found", len(demos) > 0, str(demo_dir))
if not demos:
    print("\nThere is no demo_images folder next to this script.")
    print("Unzip app_code.zip and models.zip into one folder and run this script from there.")
    sys.exit(1)
img = Image.open(demos[0]).convert("RGB")

print("\n--- normal use ---")
check_contract(predict("metadata", image=None, age=55, sex="male", location="back"), "metadata tab", False, 2)
check_contract(predict("image", image=img), "image tab", True, 1)
check_contract(predict("hybrid", image=img, age=55, sex="male", location="back"), "hybrid tab", True, 2)
check_contract(predict("hybrid", image=img), "hybrid tab, details blank", True, 2)

print("\n--- quality gate ---")
if blurred.exists():
    res = predict("image", image=Image.open(blurred).convert("RGB"))
    # only a warning about the picture itself counts, not an unrelated warning
    words = ("blur", "sharp", "quality", "retake", "focus")
    gate = [w for w in ((res or {}).get("warnings") or []) if any(k in w.lower() for k in words)]
    check("blurred image gives a quality warning", len(gate) > 0, (res or {}).get("warnings"))

print("\n--- bad input, nothing may crash ---")
bad = {
    "no image on image tab": dict(tab="image", image=None),
    "text instead of image": dict(tab="image", image="hello"),
    "10 x 10 image": dict(tab="image", image=Image.new("RGB", (10, 10), "gray")),
    "age 200": dict(tab="hybrid", image=img, age=200),
    "age 'abc'": dict(tab="hybrid", image=img, age="abc"),
    "unknown tab": dict(tab="xyz", image=img),
}
for name, kw in bad.items():
    try:
        res = predict(**kw)
        check("bad input '" + name + "' returns a dict", isinstance(res, dict) and "ok" in res,
              "ok=" + str(res.get("ok")) + " error=" + str(res.get("error")))
    except Exception:
        check("bad input '" + name + "' returns a dict", False, "it CRASHED")
        traceback.print_exc()

print()
print("RESULT:", "ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECKS FAILED: {fails}")
print("Send me this whole output.")