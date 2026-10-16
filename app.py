
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer

from hybrid_model import HybridSpamNet
from hybrid_utils import MAX_LENGTH, encode_texts


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="Sentinel AI | SMS Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------- PREMIUM UI STYLING --------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #080e1c;
        --panel: #101a2d;
        --border: #24344d;
        --cyan: #50e3e0;
        --text: #f0f5ff;
        --muted: #92a3bd;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 15% 0%, #142944 0%, transparent 42%),
            radial-gradient(ellipse at 95% 20%, #102b35 0%, transparent 30%),
            var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3, .brand {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text) !important;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 14, 28, 0.85);
    }

    [data-testid="stSidebar"] {
        background: #0b1425;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .hero {
        padding: 26px 0 20px 0;
    }

    .eyebrow {
        color: var(--cyan);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(32px, 4vw, 49px);
        font-weight: 700;
        letter-spacing: -1.8px;
        line-height: 1.1;
        margin: 0 0 14px 0;
        color: #f5f8ff;
    }

    .hero-title span {
        color: var(--cyan);
    }

    .subtitle {
        color: var(--muted);
        font-size: 15px;
        line-height: 1.7;
        max-width: 700px;
    }

    .section-card {
        background: linear-gradient(145deg, #111e32, #0d1728);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px;
        margin: 8px 0 18px 0;
    }

    .section-label {
        color: var(--cyan);
        text-transform: uppercase;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }

    .result-card {
        border-radius: 18px;
        padding: 24px;
        border: 1px solid var(--border);
        background: #101a2d;
        margin: 12px 0 18px 0;
    }

    .result-spam {
        border-color: #8f3548;
        background: linear-gradient(135deg, #301723, #171827);
    }

    .result-ham {
        border-color: #247d70;
        background: linear-gradient(135deg, #102e30, #111d2c);
    }

    .result-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .result-detail {
        color: #b8c5d9;
        font-size: 14px;
        line-height: 1.6;
    }

    .probability {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 42px;
        font-weight: 700;
        margin: 12px 0;
        color: #f5f8ff;
    }

    .small-note {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.6;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #35d6d2, #67e9d7);
        color: #06131d;
        border: none;
        border-radius: 11px;
        min-height: 48px;
        font-weight: 700;
        transition: 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        filter: brightness(1.08);
        border: none;
        box-shadow: 0 0 24px rgba(80, 227, 224, 0.18);
    }

    div[data-baseweb="select"] > div,
    textarea,
    input {
        border-radius: 10px !important;
    }

    [data-testid="stMetric"] {
        background: #101a2d;
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 15px;
    }

    hr {
        border-color: var(--border);
    }

    footer {
        visibility: hidden;
    }


    /* ---------- FINAL UI POLISH ---------- */
    .block-container {
        padding-top: 1.4rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 1500px;
    }

    .stTextArea textarea {
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
        background-color: #f1f5f9 !important;
        caret-color: #172033 !important;
        border: 1px solid #38516e !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
    }

    .stTextArea textarea::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 1 !important;
    }

    .stTextArea textarea:focus {
        border: 1px solid #50e3e0 !important;
        box-shadow: 0 0 0 2px rgba(80, 227, 224, 0.18) !important;
    }

    .metric-card {
        background: linear-gradient(145deg, #14243a, #0e192b);
        border: 1px solid #293c57;
        border-radius: 14px;
        padding: 16px 17px;
        min-height: 108px;
    }

    .metric-label {
        color: #a8bad2;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.7px;
        text-transform: uppercase;
    }

    .metric-value {
        color: #50e3e0;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 27px;
        font-weight: 700;
        margin-top: 8px;
    }

    .metric-caption {
        color: #92a3bd;
        font-size: 11px;
        margin-top: 3px;
    }

    @media (max-width: 768px) {
        .hero-title { font-size: 34px; }
        .section-card { padding: 16px; }
        .metric-card { padding: 12px; }
        .metric-value { font-size: 22px; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- MODEL LOADING --------------------

@st.cache_resource
def load_spacy():
    import spacy
    return spacy.load("en_core_web_sm")


@st.cache_resource
def load_hybrid_artifacts():
    folder = MODEL_DIR / "hybrid"

    info = json.loads(
        (folder / "hybrid_info.json").read_text(encoding="utf-8")
    )
    vocab = json.loads(
        (folder / "word_vocab.json").read_text(encoding="utf-8")
    )
    matrix = np.load(folder / "word2vec_embeddings.npy")

    checkpoint = torch.load(
        folder / "hybrid_model.pt",
        map_location="cpu",
    )

    bert_name = checkpoint.get(
        "bert_name",
        info.get("bert_name", "bert-base-uncased"),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        bert_name,
        use_fast=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = HybridSpamNet(
        embedding_matrix=matrix,
        bert_name=bert_name,
        dropout=float(checkpoint.get("dropout", 0.25)),
        freeze_bert=bool(checkpoint.get("freeze_bert", False)),
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    threshold = float(checkpoint.get("threshold", 0.5))

    return model, tokenizer, vocab, info, device, threshold


# -------------------- SESSION HISTORY --------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding:12px 0 22px 0">
            <div style="font-size:34px">🛡️</div>
            <div style="font-family:Space Grotesk;font-size:23px;font-weight:700">
                Sentinel <span style="color:#50e3e0">AI</span>
            </div>
            <div style="color:#92a3bd;font-size:12px;margin-top:5px">
                SMS THREAT DETECTION
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "Choose a section",
        ["Message Analyzer", "Prediction History", "About Project"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### System")
    st.caption("Local inference application")
    st.caption(
        f"Compute: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}"
    )
    st.caption("Dataset: SMS Spam Collection")

    st.markdown("---")
    st.markdown(
        '<div class="small-note">Research prototype · '
        'Predictions are not guaranteed to be correct.</div>',
        unsafe_allow_html=True,
    )


# -------------------- PAGE HEADER --------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-powered message protection</div>
        <div class="hero-title">Detect threats.<br><span>Trust your inbox.</span></div>
        <div class="subtitle">
            Analyze suspicious SMS messages with machine learning.
            Compare the classical baseline with the proposed hybrid
            deep-learning model.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Find available models
available = []

if (MODEL_DIR / "classical_best.joblib").exists():
    available.append("Classical TF-IDF baseline")

if (MODEL_DIR / "hybrid" / "hybrid_model.pt").exists():
    available.append("Hybrid BERT + BiLSTM + 3 GCN + GSOWOA")


if not available:
    st.warning("No trained model found in the models directory.")
    st.code(
        "python train_baselines.py\n"
        "python train_hybrid.py --freeze-bert --max-samples 500",
        language="bash",
    )
    st.stop()



# -------------------- RESEARCH METRICS DASHBOARD --------------------

st.markdown("## Research performance")
st.caption(
    "Preliminary single-run held-out test results. These are evaluation metrics, "
    "not the confidence score for the message currently being analyzed."
)

metric_cols = st.columns(4, gap="small")
dashboard_metrics = [
    ("ACCURACY", "98.74%", "Full Hybrid + GSOWOA"),
    ("SPAM PRECISION", "99.15%", "Full Hybrid + GSOWOA"),
    ("SPAM RECALL", "90.63%", "Full Hybrid + GSOWOA"),
    ("SPAM F1-SCORE", "94.69%", "Full Hybrid + GSOWOA"),
]

for col, (label, value, caption) in zip(metric_cols, dashboard_metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-caption">{caption}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("Compare recorded experiments", expanded=False):
    comparison_data = {
        "Model": [
            "Linear SVM (TF-IDF)",
            "BERT only",
            "BERT + BiLSTM",
            "BERT + 3 GCN (no BiLSTM)",
            "Full Hybrid + GSOWOA",
        ],
        "Accuracy": ["98.45%", "98.55%", "98.74%", "98.55%", "98.74%"],
        "Spam Precision": ["95.16%", "98.29%", "97.52%", "97.48%", "99.15%"],
        "Spam Recall": ["92.19%", "89.84%", "92.19%", "90.63%", "90.63%"],
        "Spam F1": ["93.65%", "93.88%", "94.78%", "93.93%", "94.69%"],
    }
    st.dataframe(comparison_data, use_container_width=True, hide_index=True)
    st.caption(
        "Preliminary results from the current experiment setup. "
        "Run repeated seeds and comparable training budgets before making "
        "a final claim that one architecture is superior."
    )

# -------------------- MESSAGE ANALYZER --------------------

if page == "Message Analyzer":

    left, right = st.columns([1.65, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-label">01 / Input</div>
                <h3 style="margin-top:0">Analyze a message</h3>
                <div class="small-note">
                    Paste the SMS you want to inspect. Your message is
                    processed locally by this application.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected = st.selectbox("Detection model", available)

        message = st.text_area(
            "SMS message",
            placeholder=(
                "Example: Congratulations! You have won a prize. "
                "Claim your reward now..."
            ),
            height=190,
            max_chars=10000,
            label_visibility="collapsed",
        )

        st.caption(f"Characters: {len(message)} / 10000")

        analyze = st.button(
            "✦  Analyze Message",
            type="primary",
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="small-note" style="margin-top:10px">
                Tip: inspect the full message, including links and
                unexpected prize or payment requests.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-label">02 / How it works</div>
                <h3 style="margin-top:0">Detection pipeline</h3>
                <p class="small-note">01 &nbsp; Text preprocessing</p>
                <p class="small-note">02 &nbsp; Model inference</p>
                <p class="small-note">03 &nbsp; Spam probability</p>
                <p class="small-note">04 &nbsp; Threshold-based decision</p>
                <hr>
                <div class="small-note">
                    The probability is a model estimate, not a guarantee
                    that a message is safe or malicious.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-card">
                <div class="section-label">03 / Research model</div>
                <h3 style="margin-top:0">Hybrid architecture</h3>
                <div class="small-note">
                    BERT semantic features<br>
                    BiLSTM sequence features<br>
                    Three graph-convolution branches<br>
                    Attention-based feature fusion
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if analyze:
        if not message.strip():
            st.warning("Please enter an SMS message first.")

        else:
            try:
                with st.spinner("Analyzing message..."):

                    if selected == "Classical TF-IDF baseline":
                        model = joblib.load(
                            MODEL_DIR / "classical_best.joblib"
                        )

                        raw_prediction = str(
                            model.predict([message])[0]
                        ).lower()

                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba([message])[0]
                            classes = list(model.classes_)

                            spam_probability = (
                                float(probabilities[classes.index("spam")])
                                if "spam" in classes
                                else 0.0
                            )
                        else:
                            spam_probability = None

                        prediction = (
                            "SPAM" if raw_prediction == "spam" else "HAM"
                        )
                        threshold_used = None

                    else:
                        model, tokenizer, vocab, info, device, threshold = (
                            load_hybrid_artifacts()
                        )

                        nlp = load_spacy()

                        encoded = encode_texts(
                            [message],
                            tokenizer,
                            nlp,
                            vocab,
                            max_length=MAX_LENGTH,
                        )

                        args = [
                            torch.as_tensor(
                                encoded["input_ids"],
                                dtype=torch.long,
                                device=device,
                            ),
                            torch.as_tensor(
                                encoded["attention_mask"],
                                dtype=torch.long,
                                device=device,
                            ),
                            torch.as_tensor(
                                encoded["word_ids"],
                                dtype=torch.long,
                                device=device,
                            ),
                            torch.as_tensor(
                                encoded["graphs"],
                                dtype=torch.uint8,
                                device=device,
                            ),
                        ]

                        with torch.no_grad():
                            logit = model(*args)
                            spam_probability = float(
                                torch.sigmoid(logit).item()
                            )

                        prediction = (
                            "SPAM"
                            if spam_probability >= threshold
                            else "HAM"
                        )
                        threshold_used = threshold

                result = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "message": message,
                    "model": selected,
                    "prediction": prediction,
                    "probability": spam_probability,
                    "threshold": threshold_used,
                }

                st.session_state.last_result = result
                st.session_state.history.insert(0, result)
                st.session_state.history = st.session_state.history[:20]

            except Exception as exc:
                st.error("Prediction failed. Check the VS Code terminal.")
                st.exception(exc)

    # Display most recent result
    result = st.session_state.last_result

    if result is not None:
        st.markdown("---")
        st.markdown("### Analysis result")

        is_spam = result["prediction"] == "SPAM"
        css_class = "result-spam" if is_spam else "result-ham"
        icon = "🚨" if is_spam else "✓"
        headline = "Potential spam detected" if is_spam else "Likely legitimate message"
        detail = (
            "Treat this message cautiously. Avoid opening unexpected links "
            "or sharing personal information."
            if is_spam
            else "The model did not classify this message as spam. "
            "This does not guarantee that it is safe."
        )

        st.markdown(
            f"""
            <div class="result-card {css_class}">
                <div class="section-label">Detection outcome</div>
                <div class="result-title">{icon} {result['prediction']}</div>
                <div style="font-size:17px;font-weight:600;margin-bottom:8px">
                    {headline}
                </div>
                <div class="result-detail">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result["probability"] is not None:
            probability = result["probability"]

            st.markdown(
                f"""
                <div class="section-card">
                    <div class="section-label">Spam probability estimate</div>
                    <div class="probability">{probability:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(max(0.0, min(1.0, probability)))

            if result["threshold"] is not None:
                st.caption(
                    f"Hybrid decision threshold: "
                    f"{result['threshold']:.2f} "
                    f"({result['threshold']:.0%})"
                )

        st.caption(f"Model: {result['model']} · Time: {result['time']}")


# -------------------- PREDICTION HISTORY --------------------

elif page == "Prediction History":
    st.markdown("## Prediction history")
    st.caption("History is kept only for this active Streamlit session.")

    history = st.session_state.history

    if not history:
        st.info("No messages analyzed in this session yet.")
    else:
        for item in history:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(
                        "**🚨 SPAM**"
                        if item["prediction"] == "SPAM"
                        else "**✓ HAM**"
                    )
                    st.write(item["message"][:300])
                    st.caption(f"{item['model']} · {item['time']}")

                with col2:
                    if item["probability"] is not None:
                        st.metric(
                            "Spam probability",
                            f"{item['probability']:.1%}",
                        )

        if st.button("Clear session history"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()


# -------------------- ABOUT PAGE --------------------

else:
    st.markdown("## About Sentinel AI")
    st.write(
        "This academic research prototype classifies pasted SMS text "
        "as spam or ham using trained machine-learning models."
    )

    st.markdown("### Available models")
    for name in available:
        st.write(f"- {name}")

    st.markdown("### Hybrid architecture")
    st.write(
        "The hybrid implementation combines BERT-based semantic features, "
        "a BiLSTM sequence branch, three graph-convolution branches, and "
        "attention-based feature fusion."
    )

    st.markdown("### Research limitations")
    st.write(
        "- Predictions can be incorrect.\n"
        "- Probability estimates may not be calibrated.\n"
        "- Results depend on the training data and selected threshold.\n"
        "- This app does not connect to a phone inbox or block SMS messages."
    )

st.markdown("---")
st.markdown(
    """
    <div class="small-note" style="text-align:center;padding:12px 0 20px 0">
        SENTINEL AI · SMS SPAM CLASSIFICATION RESEARCH PROTOTYPE<br>
        Built for academic experimentation · Not a guarantee of message safety
    </div>
    """,
    unsafe_allow_html=True,
)