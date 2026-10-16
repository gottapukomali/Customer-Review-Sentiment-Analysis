"""
Customer Review Sentiment Analysis - training script.

Dataset format (fastText style), one review per line:
    __label__1 Title: review text...     -> negative
    __label__2 Title: review text...     -> positive

Usage:
    python code.py --train train.ft.txt --test test.ft.txt --max-rows 200000
    python code.py --train reviews.txt            # no test file: auto 80/20 split
"""
import argparse
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LABEL_MAP = {"__label__1": 0, "__label__2": 1}  # 0 = negative, 1 = positive
MODEL_PATH = "sentiment_model.joblib"


def load_dataset(path, max_rows=None):
    """Read a fastText-format file into a DataFrame with columns: text, label."""
    texts, labels = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            tag, _, text = line.partition(" ")
            if tag not in LABEL_MAP:
                continue
            texts.append(text)
            labels.append(LABEL_MAP[tag])
    return pd.DataFrame({"text": texts, "label": labels})


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b[\w']+\b",
            ngram_range=(1, 2),
            min_df=3,
            max_features=300_000,
            sublinear_tf=True,
            stop_words=None,  # keep words like "not", "never" - they matter for sentiment
        )),
        ("clf", LogisticRegression(C=4.0, max_iter=1000, solver="liblinear")),
    ])


def evaluate(model, X, y, name):
    pred = model.predict(X)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y, pred):.4f}")
    print(classification_report(y, pred, target_names=["negative", "positive"], digits=4))
    print("Confusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(y, pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="path to training file")
    ap.add_argument("--test", default=None, help="optional path to test file")
    ap.add_argument("--max-rows", type=int, default=200_000, help="rows to read per file (0 = all)")
    ap.add_argument("--out", default=MODEL_PATH, help="where to save the trained model")
    args = ap.parse_args()
    max_rows = args.max_rows or None

    print("Loading training data...")
    train_df = load_dataset(args.train, max_rows)
    print(f"Train rows: {len(train_df):,} | positive share: {train_df['label'].mean():.2%}")

    if args.test:
        test_df = load_dataset(args.test, max_rows)
    else:
        train_df, test_df = train_test_split(
            train_df, test_size=0.2, random_state=42, stratify=train_df["label"]
        )
    print(f"Test rows: {len(test_df):,}")

    model = build_pipeline()
    print("Training (TF-IDF + Logistic Regression)...")
    t0 = time.time()
    model.fit(train_df["text"], train_df["label"])
    print(f"Done in {time.time() - t0:.1f}s")

    evaluate(model, test_df["text"], test_df["label"], "Test set")

    joblib.dump(model, args.out)
    print(f"\nModel saved to {args.out}")

    samples = [
        "Great product: works perfectly and arrived early. Highly recommend!",
        "Waste of money: stopped working after a week, very disappointed.",
    ]
    for s, p in zip(samples, model.predict_proba(samples)[:, 1]):
        print(f"{p:.2f} positive  <-  {s}")


if __name__ == "__main__":
    main()
