# Skin Lesion Triage Research Demo

This local Streamlit app is an educational research prototype, not a medical
device. It is not for diagnosis or treatment.

## Setup

Use Python 3.13 and install the pinned dependencies from the folder containing
`backend.py`:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Keep scikit-learn at exactly `1.6.1`; changing it can prevent the metadata
preprocessor pickle from loading. The four model files should be in the
`outputs\` folder next to `backend.py`. On first use, run while online so
Torchvision can cache the ResNet18 ImageNet weights used only by comparison
models.

## Run

From the folder containing `backend.py`:

```powershell
streamlit run app\app.py
```

The app runs locally on CPU. Metadata, image-only, and hybrid predictions use
the locked backend. Comparison models are labeled and are never used for the
final decision.

## Verify the backend

From the project root, run:

```powershell
python -u smoke_test_for_ali.py
```

The test uses the supplied `demo_images\` validation examples and should end
with `RESULT: ALL CHECKS PASSED`.
