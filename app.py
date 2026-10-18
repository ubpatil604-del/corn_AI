from pathlib import Path

import streamlit as st

from modules.vision import analyze_image_quality, decode_image, readable_label
from predict import load_model_bundle, predict_rgb


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

st.set_page_config(
    page_title="Corn-AI | Field assessment",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --forest: #173d2b;
        --leaf: #2f6b43;
        --lime: #b6d77d;
        --ink: #203128;
        --muted: #68776d;
        --line: #dce6dc;
        --canvas: #f5f8f4;
        --card: #ffffff;
        --amber: #fff6e4;
        --amber-line: #e9d39b;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 92% 0%, #e9f1e5 0, transparent 34rem),
            var(--canvas);
        color: var(--ink);
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stSidebar"] {
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--line);
        background: #fbfcfa;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] label {
        color: var(--ink) !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        border-color: #c8d6c8;
        background-color: #fff !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-testid="stSlider"] *,
    [data-testid="stSidebar"] [data-testid="stExpander"] * {
        color: var(--ink) !important;
    }
    [data-testid="stAlert"] * {
        color: #51421e !important;
    }
    [role="listbox"] *,
    [data-baseweb="popover"] [role="option"] {
        color: var(--ink) !important;
    }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    .brand {
        display: flex;
        align-items: center;
        gap: .85rem;
        margin-bottom: 1rem;
    }
    .brand-mark {
        display: grid;
        width: 48px;
        height: 48px;
        place-items: center;
        border: 1px solid #cdddc9;
        border-radius: 15px;
        background: #edf4e9;
        font-size: 1.5rem;
    }
    .brand-kicker {
        margin: 0 0 .12rem;
        color: var(--leaf);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .15em;
        text-transform: uppercase;
    }
    .brand-title {
        margin: 0;
        color: var(--forest);
        font-size: clamp(1.8rem, 3vw, 2.55rem);
        font-weight: 780;
        letter-spacing: -.045em;
        line-height: 1.08;
    }
    .brand-copy {
        max-width: 790px;
        margin: .5rem 0 1.25rem;
        color: var(--muted);
        font-size: 1rem;
    }
    .section-label {
        margin: 0 0 .55rem;
        color: var(--leaf);
        font-size: .74rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .panel-title {
        margin: 0;
        color: var(--forest);
        font-size: 1.22rem;
        font-weight: 720;
        letter-spacing: -.015em;
    }
    .panel-copy {
        margin: .35rem 0 1rem;
        color: var(--muted);
        font-size: .91rem;
    }
    .limitation-banner {
        margin: .4rem 0 1.3rem;
        padding: 1rem 1.15rem;
        border: 1px solid var(--amber-line);
        border-radius: 14px;
        background: var(--amber);
        color: #51421e;
        font-size: .9rem;
        line-height: 1.5;
    }
    .limitation-banner strong { color: #49370e; }
    .empty-state {
        display: grid;
        min-height: 310px;
        align-content: center;
        justify-items: center;
        padding: 2rem;
        border: 1px dashed #b9cbb7;
        border-radius: 20px;
        background: rgba(255,255,255,.75);
        text-align: center;
    }
    .empty-icon {
        display: grid;
        width: 62px;
        height: 62px;
        place-items: center;
        margin-bottom: .8rem;
        border-radius: 20px;
        background: #edf4e9;
        font-size: 1.9rem;
    }
    .empty-state h3 {
        margin: 0;
        color: var(--forest);
        font-size: 1.2rem;
    }
    .empty-state p {
        max-width: 480px;
        margin: .5rem 0 0;
        color: var(--muted);
        font-size: .92rem;
    }
    .workflow {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: .55rem;
        margin: .25rem 0 1.2rem;
        color: var(--muted);
        font-size: .8rem;
    }
    .workflow-step {
        padding: .34rem .65rem;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: #fff;
    }
    .workflow-step.active {
        border-color: #b7cfac;
        background: #edf5e8;
        color: var(--forest);
        font-weight: 700;
    }
    .workflow-arrow { color: #8b9a8e; }
    .quality-note {
        margin: .65rem 0 0;
        color: var(--muted);
        font-size: .79rem;
        line-height: 1.45;
    }
    .candidate-label {
        color: var(--ink);
        font-size: .9rem;
        font-weight: 650;
    }
    .score-caption {
        color: var(--muted);
        font-size: .8rem;
        text-align: right;
    }
    .footer-note {
        margin-top: 1.4rem;
        padding-top: 1rem;
        border-top: 1px solid var(--line);
        color: var(--muted);
        font-size: .8rem;
        line-height: 1.5;
    }
    div[data-testid="stMetric"] {
        padding: .8rem .9rem;
        border: 1px solid var(--line);
        border-radius: 13px;
        background: var(--card);
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    div[data-testid="stMetricValue"] { color: var(--forest); }
    div[data-testid="stProgress"] > div { border-radius: 99px; }
    @media (max-width: 700px) {
        .block-container { padding-top: 1.1rem; }
        .brand { align-items: flex-start; }
        .brand-mark { width: 42px; height: 42px; border-radius: 13px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing the selected image models…")
def cached_model_bundle(use_candidate):
    return load_model_bundle(use_candidate)


def render_candidate(candidate):
    label_column, score_column = st.columns([3, 1])
    with label_column:
        st.markdown(
            f"<div class='candidate-label'>{candidate['label']}</div>",
            unsafe_allow_html=True,
        )
    with score_column:
        st.markdown(
            f"<div class='score-caption'>{candidate['confidence']:.1%} model score</div>",
            unsafe_allow_html=True,
        )
    st.progress(
        min(max(candidate["confidence"], 0.0), 1.0),
        text=None,
    )


def render_quality_report(quality):
    st.markdown(
        "<p class='section-label'>Image quality</p>"
        "<h2 class='panel-title'>Capture conditions</h2>"
        "<p class='panel-copy'>A quality check helps interpret the image. It does not decide whether a disease is present.</p>",
        unsafe_allow_html=True,
    )
    first_row = st.columns(2)
    first_row[0].metric("Image dimensions", f"{quality['width']} × {quality['height']}")
    first_row[1].metric("Brightness", f"{quality['brightness']:.1f} / 255")
    second_row = st.columns(2)
    second_row[0].metric("Contrast", f"{quality['contrast']:.1f}")
    second_row[1].metric("Blur indicator", f"{quality['blur_score']:.1f}")
    st.caption("Blur indicator is Laplacian variance; its scale depends on image size and scene detail.")

    if quality["warnings"]:
        st.markdown("**Capture notes**")
        for warning in quality["warnings"]:
            st.warning(warning, icon="⚠️")
    else:
        st.success("No basic capture-quality warnings detected.", icon="✅")

    if quality["retake_recommended"]:
        st.info(
            "Consider retaking a clearer or better-lit image. The quality warning does not automatically block model inference.",
            icon="📷",
        )


def render_assessment(result, bundle):
    st.markdown(
        "<p class='section-label'>AI assessment</p>"
        "<h2 class='panel-title'>Corn verification & disease screening</h2>"
        "<p class='panel-copy'>The verifier runs first. Disease screening runs only when the image passes that gate.</p>",
        unsafe_allow_html=True,
    )
    status = result["status"]
    if status == "rejected":
        st.error("Not identified as a corn leaf", icon="🚫")
        st.write(result["message"])
        st.metric("Corn model score", f"{result['corn_confidence']:.1%}")
    elif status == "uncertain" and not result["top_predictions"]:
        st.warning(result["decision"], icon="⚠️")
        st.write(result["message"])
        metric_columns = st.columns(
            2 if result["corn_feature_distance"] is not None else 1
        )
        metric_columns[0].metric("Corn model score", f"{result['corn_confidence']:.1%}")
        if result["corn_feature_distance"] is not None:
            metric_columns[1].metric(
                "Feature distance",
                f"{result['corn_feature_distance']:.4f}",
                help=(
                    "Compared against the candidate detector's saved corn-feature "
                    "reference. Smaller distances are closer."
                ),
            )
    else:
        if status == "uncertain":
            st.warning("Corn leaf detected — disease result is uncertain", icon="⚠️")
            st.write(result["message"])
        else:
            st.success(f"Corn leaf detected · {result['prediction_label']}", icon="✅")

        metrics = st.columns(
            3 if result["corn_feature_distance"] is not None else 2
        )
        metrics[0].metric("Corn model score", f"{result['corn_confidence']:.1%}")
        metrics[1].metric("Top disease model score", f"{result['confidence']:.1%}")
        if result["corn_feature_distance"] is not None:
            metrics[2].metric(
                "Feature distance",
                f"{result['corn_feature_distance']:.4f}",
                help="Smaller values are closer to the candidate detector's corn reference.",
            )
        st.markdown("**Disease candidates**")
        for candidate in result["top_predictions"]:
            render_candidate(candidate)

    st.caption(
        "Model scores are not calibrated probabilities or guarantees of correctness. "
        "This image-level classifier does not locate lesions or estimate disease severity."
    )


def main():
    st.markdown(
        """
        <div class="brand">
          <div class="brand-mark">🌽</div>
          <div>
            <p class="brand-kicker">Field image assessment</p>
            <h1 class="brand-title">Corn-AI</h1>
          </div>
        </div>
        <p class="brand-copy">
          Review leaf-image quality, verify whether the image resembles corn,
          and screen supported corn disease classes.
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Assessment settings")
        st.caption("Choose the model bundle and review its decision cutoffs.")
        candidate_available = (
            (MODEL_DIR / "corn_candidate_config.json").is_file()
            and (MODEL_DIR / "disease_candidate_config.json").is_file()
        )
        model_options = ["Active models"]
        if candidate_available:
            model_options.append("Candidate models · not promoted")
        selected_model = st.selectbox("Model bundle", model_options)
        use_candidate = selected_model.startswith("Candidate")

        try:
            bundle = cached_model_bundle(use_candidate)
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
            st.error(f"Could not load the selected models: {error}")
            st.stop()

        st.markdown("---")
        st.markdown("**Corn verification cutoff**")
        corn_threshold = st.slider(
            "Corn cutoff",
            min_value=0.0,
            max_value=1.0,
            value=float(bundle["corn_threshold"]),
            step=0.01,
            label_visibility="collapsed",
            help="Images below this detector score are not passed to disease classification.",
        )
        st.markdown("**Disease abstention cutoff**")
        disease_threshold = st.slider(
            "Disease cutoff",
            min_value=0.0,
            max_value=1.0,
            value=float(bundle["disease_threshold"]),
            step=0.01,
            label_visibility="collapsed",
            help="A top disease score below this cutoff is reported as uncertain.",
        )
        st.caption("Cutoffs change the accept/abstain decision; they do not calibrate the scores.")

        if not bundle["thresholds_calibrated"]:
            st.warning(
                "These active-model cutoffs are legacy defaults, not validation-calibrated.",
                icon="⚠️",
            )
        if not bundle["open_set_guard_available"]:
            st.warning(
                "The active detector has no feature-distance open-set guard. A finite negative set cannot cover every non-corn image.",
                icon="⚠️",
            )

        with st.expander("About the model"):
            st.write(f"Selected bundle: **{bundle['bundle_name']}**")
            st.write(
                "**Supported disease labels:** "
                + ", ".join(readable_label(name) for name in bundle["disease_classes"])
            )
            st.write(
                "Images are processed by the hosted app when deployed. Avoid uploading sensitive or identifiable images."
            )

    disease_classes = {
        name.casefold().replace("_", " ") for name in bundle["disease_classes"]
    }
    supports_nlb = any("northern leaf blight" in name for name in disease_classes)
    if not supports_nlb:
        st.markdown(
            """
            <div class="limitation-banner">
              <strong>Model scope:</strong> Northern leaf blight is not included in the
              selected disease model. Current supported classes are shown in the sidebar.
              This is an experimental screening aid, not a field-validated diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="workflow" aria-label="Assessment steps">
          <span class="workflow-step active">01 · Image quality</span>
          <span class="workflow-arrow">→</span>
          <span class="workflow-step">02 · Corn verification</span>
          <span class="workflow-arrow">→</span>
          <span class="workflow-step">03 · Disease screening</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploader_column, helper_column = st.columns([1.4, 1])
    with uploader_column:
        uploaded = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png", "webp"],
            help="For best results, use a sharp, well-lit image with the leaf clearly visible.",
        )
    with helper_column:
        st.markdown(
            """
            <div class="quality-note">
              <strong>Good capture tips</strong><br>
              Keep the leaf in focus · avoid harsh glare and deep shadow ·
              include enough leaf detail for visual inspection.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded is None:
        st.markdown(
            """
            <div class="empty-state">
              <div class="empty-icon">↥</div>
              <h3>Start with a leaf photo</h3>
              <p>Upload a JPG, PNG, or WEBP image. We’ll show image-quality indicators and run the selected models on the image.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            image_rgb = decode_image(uploaded.getvalue(), uploaded.name)
        except (ValueError, OSError) as error:
            st.error(f"Unable to open this image: {error}")
            st.stop()

        st.markdown("---")
        image_column, report_column = st.columns([1, 1], gap="large")
        with image_column:
            st.markdown(
                "<p class='section-label'>Uploaded image</p>"
                "<h2 class='panel-title'>Visual review</h2>",
                unsafe_allow_html=True,
            )
            st.image(
                image_rgb,
                caption=uploaded.name,
                width="stretch",
            )
            render_quality_report(analyze_image_quality(image_rgb))

        with report_column:
            try:
                result = predict_rgb(
                    image_rgb,
                    {
                        **bundle,
                        "corn_threshold": corn_threshold,
                        "disease_threshold": disease_threshold,
                    },
                )
            except (ValueError, OSError, RuntimeError) as error:
                st.error(f"Prediction failed: {error}")
                st.stop()
            render_assessment(result, bundle)

    st.markdown(
        """
        <div class="footer-note">
          <strong>Use responsibly.</strong> This model only classifies the image as a whole;
          it does not mark lesion locations or measure disease severity. Confirm important
          decisions with field inspection or a qualified plant-health expert. Thresholds
          and scores are experimental, and performance may differ on local varieties,
          cameras, lighting, and field backgrounds.
        </div>
        """,
        unsafe_allow_html=True,
    )


main()
