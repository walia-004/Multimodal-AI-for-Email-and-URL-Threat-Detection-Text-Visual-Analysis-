# train_url.py

import pandas as pd
import joblib
import re
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix
)

# -------------------------------
# Feature extraction
# -------------------------------
def extract_features(url):
    return [
        len(url),
        url.count('.'),
        url.count('@'),
        url.count('-'),
        sum(c.isdigit() for c in url),
        int("https" in url.lower()),
        int(bool(re.search(r"login|verify|secure|update", url.lower())))
    ]

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("data/malicious_phish.csv")

df["label_binary"] = df["type"].apply(
    lambda x: 0 if x == "benign" else 1
)

X = df["url"].apply(extract_features).tolist()
y = df["label_binary"]

print("\n📊 Class distribution (overall dataset):")
print(y.value_counts(normalize=True))

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Base model
# -------------------------------
base_model = RandomForestClassifier(
    random_state=42,
    n_jobs=1
)

# -------------------------------
# Hyperparameter search space
# -------------------------------
param_dist = {
    "n_estimators": [100, 150, 200],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"],
    "class_weight": ["balanced"]
}

# -------------------------------
# Hyperparameter tuning
# -------------------------------
print("\n🚀 Starting hyperparameter tuning...")

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=10,
    scoring="f1",
    cv=3,
    verbose=2,          # <-- shows training logs
    random_state=42,
    n_jobs=1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\n✅ Best Parameters Found:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

# -------------------------------
# Evaluation
# -------------------------------
y_pred = best_model.predict(X_test)

f1 = f1_score(y_test, y_pred)

print("\n📊 URL Model Evaluation:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(f"\n🎯 Final F1 Score: {f1:.4f}")

# -------------------------------
# Conditional save
# -------------------------------
F1_THRESHOLD = 0.90

if f1 >= F1_THRESHOLD:
    joblib.dump(best_model, "models/url_model.pkl")
    print(f"\n✅ Model saved (F1 ≥ {F1_THRESHOLD})")
else:
    print(f"\n❌ Model NOT saved (F1 < {F1_THRESHOLD})")
    print("👉 Try increasing n_iter or adding new features")