"""Streamlit web app for tweet sentiment analysis.

Run:  streamlit run app.py
"""
import json
import os

import pandas as pd
import streamlit as st
import tensorflow as tf

from utils import clean_text

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.keras")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
HISTORY_PNG = os.path.join(MODELS_DIR, "training_history.png")
MAX_TOKENS = 50  # must match MAX_LEN in train.py

POS_COLOR = "#16a34a"
NEG_COLOR = "#dc2626"

CSS = f"""
<style>
.block-container {{ max-width: 820px; padding-top: 2.5rem; }}
.app-header h1 {{ font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }}
.app-header p {{ opacity: 0.65; margin-top: 0; }}
.result-card {{
    border: 1px solid rgba(128,128,128,0.25);
    border-left: 6px solid var(--accent);
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin: 1rem 0 0.75rem;
    background: rgba(128,128,128,0.06);
}}
.result-label {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.6; }}
.result-value {{ font-size: 1.8rem; font-weight: 700; color: var(--accent); line-height: 1.3; }}
.result-sub {{ font-size: 0.95rem; opacity: 0.75; }}
.split-bar {{
    display: flex; height: 10px; border-radius: 5px; overflow: hidden; margin: 0.9rem 0 0.35rem;
}}
.split-legend {{ display: flex; justify-content: space-between; font-size: 0.82rem; opacity: 0.75; }}
</style>
"""


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict(model, texts):
    cleaned = [clean_text(t) for t in texts]
    return model.predict(tf.constant(cleaned), verbose=0).ravel()


def result_card(prob):
    positive = prob >= 0.5
    label = "Positive" if positive else "Negative"
    accent = POS_COLOR if positive else NEG_COLOR
    confidence = prob if positive else 1 - prob
    strength = "Strong" if confidence >= 0.85 else "Moderate" if confidence >= 0.65 else "Weak"
    pos_pct, neg_pct = prob * 100, (1 - prob) * 100
    return f"""
<div class="result-card" style="--accent:{accent}">
  <div class="result-label">Predicted sentiment</div>
  <div class="result-value">{label}</div>
  <div class="result-sub">{strength} signal &middot; {confidence:.1%} confidence</div>
  <div class="split-bar">
    <div style="width:{neg_pct:.1f}%; background:{NEG_COLOR}"></div>
    <div style="width:{pos_pct:.1f}%; background:{POS_COLOR}"></div>
  </div>
  <div class="split-legend"><span>Negative {neg_pct:.1f}%</span><span>Positive {pos_pct:.1f}%</span></div>
</div>
"""


st.set_page_config(page_title="Tweet Sentiment Analyzer", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    """
<div class="app-header">
  <h1>Tweet Sentiment Analyzer</h1>
  <p>Classifies a tweet as positive or negative using a Bidirectional LSTM trained on 1.6M-tweet Sentiment140 data.</p>
</div>
""",
    unsafe_allow_html=True,
)

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

model = load_model()

tab_single, tab_batch, tab_about = st.tabs(["Analyze", "Batch analysis", "Model details"])

with tab_single:
    with st.form("analyze_form", border=False):
        text = st.text_area(
            "Tweet text",
            height=140,
            placeholder="Paste a post from X.com or type your own text...",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Analyze sentiment", type="primary", use_container_width=True)

    if submitted:
        if not text.strip():
            st.warning("Enter some text to analyze.")
        else:
            prob = float(predict(model, [text])[0])
            st.markdown(result_card(prob), unsafe_allow_html=True)

            cleaned = clean_text(text)
            n_tokens = len(cleaned.split())
            if n_tokens > MAX_TOKENS:
                st.caption(f"Note: only the first {MAX_TOKENS} of {n_tokens} words were used by the model.")
            with st.expander("What the model saw"):
                st.code(cleaned or "(empty after cleaning)", language=None, wrap_lines=True)

with tab_batch:
    st.write("Upload a CSV file with a `text` column to classify many tweets at once.")
    uploaded = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("The CSV needs a column named `text`.")
        else:
            with st.spinner("Classifying..."):
                probs = predict(model, df["text"].astype(str).tolist())
            df["p_positive"] = probs.round(4)
            df["sentiment"] = ["positive" if p >= 0.5 else "negative" for p in probs]

            n_pos = int((df["sentiment"] == "positive").sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Tweets", f"{len(df):,}")
            c2.metric("Positive", f"{n_pos:,}  ({n_pos / len(df):.0%})")
            c3.metric("Negative", f"{len(df) - n_pos:,}  ({1 - n_pos / len(df):.0%})")

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "p_positive": st.column_config.ProgressColumn(
                        "P(positive)", min_value=0.0, max_value=1.0, format="%.2f"
                    )
                },
            )
            st.download_button(
                "Download results as CSV", df.to_csv(index=False), "sentiment_results.csv", "text/csv"
            )

with tab_about:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        c1, c2, c3 = st.columns(3)
        c1.metric("Test accuracy", f"{metrics['test_accuracy']:.1%}")
        c2.metric("Hand-labelled test", f"{metrics['manual_test_accuracy']:.1%}")
        c3.metric("Training tweets", f"{metrics['train_samples']:,}")
    if os.path.exists(HISTORY_PNG):
        st.image(HISTORY_PNG, caption="Accuracy and loss per epoch")

    st.subheader("Pipeline")
    st.markdown(
        """
| Stage | Details |
|---|---|
| Text cleaning | lowercase, replace URLs and @mentions, strip symbols, shorten repeated letters |
| Vectorization | 20,000-word vocabulary, sequences padded/truncated to 50 tokens |
| Embedding | 128 dimensions, learned during training |
| Bidirectional LSTM | 64 units per direction |
| Classifier | Dense (64, ReLU) with dropout, then Dense (1, sigmoid) |
"""
    )
