import io
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import predict


CLASS_NAMES = {
    "akiec": "actinic keratosis",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic nevus",
    "vasc": "vascular lesion",
}
CLASS_ORDER = list(CLASS_NAMES)
DECISION_COLORS = {
    "referred": ("#8d2635", "#fff1f2"),
    "deferred": ("#855b00", "#fff7e0"),
    "not referred": ("#216b4b", "#e9f7ef"),
}
DISCLAIMER = (
    "Educational research prototype. NOT a medical device. "
    "Not for diagnosis or treatment."
)

st.set_page_config(page_title="Lesion Triage Research Demo", page_icon="🔬", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; color: #182536; --text-color: #182536; }
    .block-container { max-width: 1220px; padding-top: 2rem; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp p:not(.decision-title), .stApp label,
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stFileUploaderDropzone"] {
      color: #182536 !important;
    }
    .stApp [data-testid="stAlertContainer"] { background: transparent !important; }
    .stApp [data-testid="stAlertContentError"] {
      background: #fff1f2; border: 1px solid #e6aeb5; border-radius: 8px;
    }
    .stApp [data-testid="stAlertContentError"] p { color: #7a2430 !important; }
    .stApp [data-testid="stAlertContentWarning"] {
      background: #fff7e0; border: 1px solid #e8c66d; border-radius: 8px;
    }
    .stApp [data-testid="stAlertContentWarning"] p { color: #614b12 !important; }
    .stApp [data-testid="stPlotlyChart"] svg text { fill: #40536a !important; }
    .app-kicker { color: #53647b; font-size: .82rem; font-weight: 700;
                  letter-spacing: .12em; text-transform: uppercase; }
    .stApp .app-kicker { color: #53647b !important; }
    .disclaimer { background: #fff3d6; border: 1px solid #e8c66d;
                  border-radius: 10px; color: #614b12; font-weight: 700;
                  padding: .8rem 1rem; margin: .6rem 0 1.2rem; }
    .decision-card { border-radius: 12px; padding: 1rem 1.2rem;
                     border-left: 6px solid; margin: .6rem 0 1rem; }
    .decision-title { font-size: 1.35rem; font-weight: 750; margin: 0; }
    .decision-reason { margin: .35rem 0 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-kicker">Local research prototype · CPU inference</div>', unsafe_allow_html=True)
st.title("Skin lesion triage")
st.write("Explore the locked research models and their comparison baselines.")


def input_details(prefix):
    age_col, sex_col, location_col = st.columns(3)
    with age_col:
        has_age = st.checkbox("Provide age", value=False, key=prefix + "_has_age")
        age = (
            st.number_input("Age", min_value=0, max_value=120, value=50, step=1, key=prefix + "_age")
            if has_age
            else None
        )
    with sex_col:
        sex_label = st.selectbox(
            "Sex",
            ["Unknown", "Female", "Male"],
            key=prefix + "_sex",
        )
    with location_col:
        location_label = st.selectbox(
            "Body location",
            [
                "Unknown",
                "Abdomen",
                "Acral",
                "Back",
                "Chest",
                "Ear",
                "Face",
                "Foot",
                "Genital",
                "Hand",
                "Lower extremity",
                "Neck",
                "Scalp",
                "Trunk",
                "Upper extremity",
            ],
            key=prefix + "_location",
        )
    return age, sex_label.lower() if sex_label != "Unknown" else None, (
        location_label.lower() if location_label != "Unknown" else None
    )


def read_uploaded_image(upload):
    try:
        image = Image.open(io.BytesIO(upload.getvalue()))
        image.load()
        return image.convert("RGB"), None
    except (UnidentifiedImageError, OSError, ValueError):
        return None, "That file could not be opened as an image. Please choose a valid image file."


def show_result(result, image=None, tab_key=""):
    if not result.get("ok"):
        st.error(result.get("error") or "Prediction could not be completed.")
        return

    for warning in result.get("warnings", []):
        st.warning(warning)

    final = result["final"]
    title_col, confidence_col, time_col = st.columns([2, 1, 1])
    title_col.subheader(CLASS_NAMES.get(final["pred"], final["pred"]).title())
    confidence_col.metric("Confidence", f"{final['confidence']:.1%}")
    time_col.metric("Time taken", f"{final['time_ms']:.1f} ms")
    st.caption(f"Headline model: {final['model']}")

    decision = final["decision"]
    border, background = DECISION_COLORS.get(decision, ("#53647b", "#eef2f6"))
    st.markdown(
        f"""
        <div class="decision-card" style="border-color:{border};background:{background}">
          <p class="decision-title" style="color:{border}">Decision: {decision.title()}</p>
          <p class="decision-reason"><b>Reason:</b> {final['decision_reason']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if image is not None:
        original_col, heatmap_col = st.columns(2)
        with original_col:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with heatmap_col:
            if final.get("heatmap") is not None:
                st.image(final["heatmap"], caption="Grad-CAM heatmap", use_container_width=True)
            else:
                st.info("A Grad-CAM heatmap is not available for this result.")

    st.markdown("#### Class probabilities")
    probabilities = final["probs"]
    labels = [CLASS_NAMES[name] for name in CLASS_ORDER]
    figure = go.Figure(
        go.Bar(
            x=[probabilities[name] for name in CLASS_ORDER],
            y=labels,
            orientation="h",
            marker_color="#4b78a8",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    figure.update_layout(
        height=330,
        margin=dict(l=12, r=12, t=10, b=10),
        xaxis=dict(title="Probability", range=[0, 1], tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, use_container_width=True, key=tab_key + "_probabilities")

    others = result.get("others", [])
    st.markdown("#### Comparison only, not used for the decision")
    if others:
        rows = []
        for other in others:
            rows.append(
                {
                    "Model": other["model"],
                    "Predicted class": CLASS_NAMES.get(other["pred"], other["pred"]),
                    "Confidence": f"{other['confidence']:.1%}",
                    "Decision": other["decision"],
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No comparison results are available.")


def render_tab(tab, prefix, needs_image, needs_details=False):
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)
    uploaded = None
    if needs_image:
        uploaded = st.file_uploader(
            "Upload a lesion image",
            key=prefix + "_upload",
        )
    age = sex = location = None
    if needs_details:
        st.markdown("#### Optional patient details")
        age, sex, location = input_details(prefix)
    elif tab == "metadata":
        st.markdown("#### Patient details")
        age, sex, location = input_details(prefix)

    if st.button("Run prediction", type="primary", key=prefix + "_run"):
        image, friendly_error = (None, None)
        if needs_image:
            if uploaded is None:
                friendly_error = "Please upload an image to run this prediction."
            else:
                image, friendly_error = read_uploaded_image(uploaded)

        if friendly_error:
            st.session_state[prefix + "_error"] = friendly_error
            st.session_state.pop(prefix + "_result", None)
            st.session_state.pop(prefix + "_image", None)
        else:
            st.session_state.pop(prefix + "_error", None)
            with st.spinner("Running CPU prediction..."):
                result = predict(tab, image=image, age=age, sex=sex, location=location)
            st.session_state[prefix + "_result"] = result
            st.session_state[prefix + "_image"] = image

    error = st.session_state.get(prefix + "_error")
    if error:
        st.error(error)
    else:
        result = st.session_state.get(prefix + "_result")
        if result is not None:
            show_result(result, st.session_state.get(prefix + "_image"), prefix)
        else:
            st.info("Enter the available details and select **Run prediction** to see a result.")


metadata_tab, image_tab, hybrid_tab = st.tabs(["1 · Metadata", "2 · Image", "3 · Hybrid"])
with metadata_tab:
    render_tab("metadata", "metadata", needs_image=False)
with image_tab:
    render_tab("image", "image", needs_image=True)
with hybrid_tab:
    render_tab("hybrid", "hybrid", needs_image=True, needs_details=True)
