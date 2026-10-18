import streamlit as st
from pathlib import Path

from modules.vision import analyze_image_quality, decode_image, readable_label
from predict import load_model_bundle, predict_rgb


st.set_page_config(
    page_title="Corn-AI | Leaf health",
    page_icon="🌽",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f8f1; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(115deg, #214c34, #528447);
        color: white;
        margin-bottom: 1.3rem;
    }
    .hero h1 { margin: 0; color: white; }
    .hero p { margin: .45rem 0 0; color: #edf5e9; }
    .result-card {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: white;
        border: 1px solid #dce7d5;
    }
    </style>
    <div class="hero">
      <h1>Corn-AI Leaf Health</h1>
      <p>Image quality review, corn-leaf verification, and disease classification.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the selected models...")
def cached_model_bundle(use_candidate):
    return load_model_bundle(use_candidate)


candidate_available = (
    (Path(__file__).resolve().parent / "models" / "corn_candidate_config.json").is_file()
    and (Path(__file__).resolve().parent / "models" / "disease_candidate_config.json").is_file()
)
model_options = ["Active models"]
if candidate_available:
    model_options.append("Candidate models (not promoted)")
selected_model = st.sidebar.selectbox("Model set", model_options)
use_candidate = selected_model.startswith("Candidate")

with st.sidebar:
    st.subheader("Decision cutoffs")
    st.caption("Cutoffs affect whether a result is accepted; they are not probabilities.")
    try:
        bundle = cached_model_bundle(use_candidate)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        st.error(f"Could not load models: {error}")
        st.stop()

    if not bundle["thresholds_calibrated"]:
        st.warning(
            "The active models have no saved validation cutoffs. These are legacy "
            "fallbacks; evaluate and calibrate before field use."
        )
    if not bundle["open_set_guard_available"]:
        st.warning(
            "This model set has no validation-calibrated feature-distance guard. "
            "Corn verification therefore remains limited by its finite negative examples."
        )
    corn_threshold = st.slider(
        "Corn verification cutoff",
        min_value=0.0,
        max_value=1.0,
        value=float(bundle["corn_threshold"]),
        step=0.01,
    )
    disease_threshold = st.slider(
        "Disease confidence cutoff",
        min_value=0.0,
        max_value=1.0,
        value=float(bundle["disease_threshold"]),
        step=0.01,
    )
    st.caption(f"Selected model set: {bundle['bundle_name']}")

active_classes = {name.casefold().replace("_", " ") for name in bundle["disease_classes"]}
if not any("northern leaf blight" in name for name in active_classes):
    st.warning(
        "The selected disease model has no Northern leaf blight output. The current "
        "dataset/model supports only: "
        + ", ".join(readable_label(name) for name in bundle["disease_classes"])
        + ". Add the missing class, retrain, independently evaluate, then select the candidate."
    )

uploaded = st.file_uploader(
    "Upload a corn-leaf image",
    type=["jpg", "jpeg", "png", "webp"],
    help="A clear, well-lit close-up usually provides the best input.",
)

if uploaded is None:
    st.info("Upload an image to inspect its quality and run model inference.")
else:
    try:
        image_rgb = decode_image(uploaded.getvalue(), uploaded.name)
    except (ValueError, OSError) as error:
        st.error(str(error))
        st.stop()

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.image(image_rgb, caption=uploaded.name, use_container_width=True)
        st.subheader("Image quality")
        quality = analyze_image_quality(image_rgb)
        metric_columns = st.columns(2)
        metric_columns[0].metric("Image size", f"{quality['width']} × {quality['height']}")
        metric_columns[1].metric("Brightness", f"{quality['brightness']:.1f} / 255")
        metric_columns[0].metric("Contrast", f"{quality['contrast']:.1f}")
        metric_columns[1].metric("Blur score", f"{quality['blur_score']:.1f}")
        st.caption("Blur score is Laplacian variance; higher values generally indicate more edge detail.")
        if quality["warnings"]:
            for warning in quality["warnings"]:
                st.warning(warning)
        else:
            st.success("No basic image-quality warnings detected.")
        if quality["retake_recommended"]:
            st.info("A retake may help. Quality warnings do not automatically block inference.")

    with right:
        st.subheader("Model assessment")
        try:
            result = predict_rgb(image_rgb, bundle)
        except (ValueError, OSError, RuntimeError) as error:
            st.error(f"Prediction failed: {error}")
            st.stop()

        if result["status"] == "rejected":
            st.error("Not a corn leaf")
            st.write(result["message"])
            st.metric("Corn score", f"{result['corn_confidence']:.1%}")
        elif result["status"] == "uncertain" and not result["top_predictions"]:
            st.warning(result["decision"])
            st.write(result["message"])
            st.metric("Corn score", f"{result['corn_confidence']:.1%}")
            if result["corn_feature_distance"] is not None:
                st.metric(
                    "Corn feature distance",
                    f"{result['corn_feature_distance']:.4f} / "
                    f"{bundle['feature_distance_threshold']:.4f}",
                )
        else:
            if result["status"] == "uncertain":
                st.warning(result["message"])
            else:
                st.success(result["message"])
            result_columns = st.columns(3 if result["corn_feature_distance"] is not None else 2)
            result_columns[0].metric("Corn score", f"{result['corn_confidence']:.1%}")
            result_columns[1].metric(
                "Top disease score",
                f"{result['confidence']:.1%}",
            )
            if result["corn_feature_distance"] is not None:
                result_columns[2].metric(
                    "Corn feature distance",
                    f"{result['corn_feature_distance']:.4f}",
                )
            if result["top_predictions"]:
                st.markdown("**Top disease candidates**")
                for candidate in result["top_predictions"]:
                    st.progress(
                        min(max(candidate["confidence"], 0.0), 1.0),
                        text=f"{candidate['label']} — {candidate['confidence']:.1%}",
                    )

        st.caption(
            "Scores are model outputs, not guaranteed probabilities of correctness. "
            "A classifier assigns an image-level label; it does not locate lesions "
            "or estimate affected leaf area."
        )
