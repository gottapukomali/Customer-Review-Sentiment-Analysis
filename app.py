"""
Streamlit app for customer review sentiment analysis.
Run:  streamlit run app.py
Requires sentiment_model.joblib (created by:  python code.py --train <file>)
"""
import io
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "sentiment_model.joblib"

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="💬", layout="centered")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def label_from_prob(p):
    return "Positive 😊" if p >= 0.5 else "Negative 😞"


st.title("💬 Customer Review Sentiment Analysis")
st.caption("TF-IDF + Logistic Regression trained on product reviews")

model = load_model()
if model is None:
    st.error(f"Model file '{MODEL_PATH}' not found. Run `python code.py --train <your_file>` first.")
    st.stop()

tab_single, tab_batch = st.tabs(["Single review", "Batch (CSV / TXT)"])

with tab_single:
    review = st.text_area("Enter a review", height=150,
                          placeholder="e.g. Great product: works perfectly, arrived early!")
    if st.button("Analyze", type="primary"):
        if not review.strip():
            st.warning("Please enter some text.")
        else:
            prob = float(model.predict_proba([review])[0, 1])
            conf = prob if prob >= 0.5 else 1 - prob
            st.subheader(label_from_prob(prob))
            st.progress(prob, text=f"Positive probability: {prob:.1%}")
            st.write(f"Confidence: **{conf:.1%}**")

with tab_batch:
    st.write("Upload a **CSV** with a `review` column (or any text column), or a **TXT** with one review per line.")
    up = st.file_uploader("Upload file", type=["csv", "txt"])
    if up is not None:
        if up.name.lower().endswith(".csv"):
            df = pd.read_csv(up)
            col = "review" if "review" in df.columns else st.selectbox("Text column", df.columns)
            texts = df[col].astype(str)
        else:
            lines = io.StringIO(up.getvalue().decode("utf-8", errors="ignore")).read().splitlines()
            df = pd.DataFrame({"review": [l for l in lines if l.strip()]})
            texts = df["review"]

        probs = model.predict_proba(texts)[:, 1]
        df["positive_prob"] = probs.round(4)
        df["sentiment"] = ["Positive" if p >= 0.5 else "Negative" for p in probs]

        c1, c2 = st.columns(2)
        c1.metric("Positive", int((df["sentiment"] == "Positive").sum()))
        c2.metric("Negative", int((df["sentiment"] == "Negative").sum()))
        st.bar_chart(df["sentiment"].value_counts())
        st.dataframe(df, use_container_width=True)
        st.download_button("Download results (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "sentiment_results.csv", "text/csv")