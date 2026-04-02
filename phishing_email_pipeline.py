# phishing_email_pipeline.py
# --------------------------------------------------
# Email phishing detection with calibrated LinearSVC
# Provides predict_proba() for multimodal fusion
# --------------------------------------------------

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

RANDOM_STATE = 42


# ==============================
# 1. LOAD & CLEAN DATA
# ==============================

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    print("Dataset shape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    df = df.dropna(subset=["text_combined", "label"]).copy()
    df["text_combined"] = df["text_combined"].astype(str)
    df["label"] = df["label"].astype(int)

    before = len(df)
    df = df.drop_duplicates(subset=["text_combined", "label"])
    print(f"Dropped {before - len(df)} duplicate rows")

    return df


# ==============================
# 2. BUILD PIPELINE
# ==============================

def build_pipeline():
    base_svm = LinearSVC()

    calibrated_svm = CalibratedClassifierCV(
        estimator=base_svm,
        method="sigmoid",   # best for small/medium datasets
        cv=3
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_df=0.9,
                min_df=5
            )),
            ("clf", calibrated_svm)
        ]
    )
    return pipeline


# ==============================
# 3. HYPERPARAMETER TUNING
# ==============================

def tune_hyperparameters(X_train, y_train, pipeline: Pipeline) -> GridSearchCV:
    param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [3, 5, 10],
    "tfidf__max_df": [0.9, 0.95],
    "clf__estimator__C": [0.5, 1.0, 2.0],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,   # IMPORTANT for Windows
        verbose=2
    )

    grid.fit(X_train, y_train)

    print("\nBest parameters:", grid.best_params_)
    print("Best CV F1-score:", grid.best_score_)

    return grid


# ==============================
# 4. EVALUATION
# ==============================

def evaluate_model(model, X_test, y_test) -> None:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== Test set performance ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# ==============================
# 5. SAVE / LOAD
# ==============================

def save_model(model, path="phishing_email_model.joblib") -> None:
    joblib.dump(model, path)
    print(f"\nModel saved to: {path}")


def load_model(path="phishing_email_model.joblib"):
    return joblib.load(path)


def predict_email(text: str, model) -> tuple:
    """
    Returns:
    - predicted class (0/1)
    - phishing probability
    """
    prob = model.predict_proba([text])[0][1]
    label = int(prob >= 0.5)
    return label, prob


# ==============================
# 6. MAIN
# ==============================

def main():
    csv_path = "data/phishing_email.csv"  # update if needed

    print("Loading data...")
    df = load_and_clean_data(csv_path)

    X = df["text_combined"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("\nBuilding pipeline...")
    pipeline = build_pipeline()

    print("\nRunning GridSearchCV...")
    grid = tune_hyperparameters(X_train, y_train, pipeline)

    print("\nEvaluating best model...")
    best_model = grid.best_estimator_
    evaluate_model(best_model, X_test, y_test)

    save_model(best_model)

    # Demo inference
    sample_email = (
        "Your account has been suspended. "
        "Click the link below to verify your information."
    )

    label, prob = predict_email(sample_email, best_model)
    print("\nSample Email Prediction")
    print("Label:", label, "(1 = phishing)")
    print("Phishing probability:", round(prob, 4))


if __name__ == "__main__":
    main()