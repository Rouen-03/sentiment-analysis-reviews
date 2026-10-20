"""Streamlit app: sentiment analysis for product reviews.

Run from the project root:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import joblib
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import preprocess_text

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "sentiment_pipeline.joblib"
CLASS_ORDER = ["negative", "neutral", "positive"]

st.set_page_config(page_title="Review Sentiment Analyzer", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("Review Sentiment Analyzer")
st.markdown(
    "A TF-IDF (unigrams + bigrams) + Logistic Regression model trained on "
    "10,254 Amazon musical-instrument reviews, with stratified splitting and "
    "class-weight balancing. Enter a review and see how the model reads it."
)

st.write("**Try an example:**")
examples = {
    "Glowing": "This pedal is amazing! Best purchase I've made all year, sounds incredible.",
    "Angry": "Total waste of money. It doesn't work at all and broke after two days.",
    "Meh": "It's an okay cable, does the job but nothing special.",
    "Mixed": "The tuner works sometimes, but the display is hard to read in daylight.",
}

for col, (name, text) in zip(st.columns(4), examples.items()):
    if col.button(name):
        st.session_state["review_text"] = text

user_text = st.text_area(
    "Your review",
    key="review_text",
    height=150,
    placeholder="e.g. Great sound quality, easy to install, arrived quickly...",
)

if st.button("Analyze sentiment", type="primary"):
    if not user_text.strip():
        st.warning("Please enter a review first.")
    else:
        clean = preprocess_text(user_text)
        if not clean:
            st.warning("No analyzable words after preprocessing - try a longer review.")
        else:
            proba = model.predict_proba([clean])[0]
            pred = model.classes_[proba.argmax()]

            st.subheader(f"Prediction: {pred}")
            st.markdown("**Class probabilities**")
            for cls, p in zip(CLASS_ORDER, proba):
                st.progress(float(p), text=f"{cls}: {p:.1%}")

            with st.expander("See how your review was preprocessed"):
                st.code(clean)

st.caption(
    "Model: TF-IDF (1-2 grams, 20k features) + Logistic Regression (class_weight=balanced) "
    "| Data: Amazon Musical Instruments Reviews | Test macro-F1 0.58"
)