# train_url_cached.py (Deep URL Embeddings with caching)

import os
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -------------------------------
# Directories
# -------------------------------
os.makedirs("cache", exist_ok=True)
os.makedirs("models", exist_ok=True)

EMBED_CACHE_TRAIN = "cache/X_train_emb.npy"
EMBED_CACHE_TEST = "cache/X_test_emb.npy"

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("data/malicious_phish.csv")
df["label_binary"] = df["type"].apply(lambda x: 0 if x == "benign" else 1)

urls = df["url"].astype(str).tolist()
labels = df["label_binary"].values

print("\n📊 Class distribution:")
print(pd.Series(labels).value_counts(normalize=True))

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    urls,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# -------------------------------
# Load embedding model
# -------------------------------
print("\n🔄 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Embedding cache function
# -------------------------------
def get_embeddings(urls, cache_path):
    if os.path.exists(cache_path):
        print(f"\n💾 Loading cached embeddings from {cache_path}")
        return np.load(cache_path)
    else:
        print(f"\n🧠 Generating embeddings and saving to {cache_path} ...")
        emb = embedder.encode(
            urls,
            batch_size=64,
            show_progress_bar=True
        )
        np.save(cache_path, emb)
        return emb

X_train_emb = get_embeddings(X_train, EMBED_CACHE_TRAIN)
X_test_emb = get_embeddings(X_test, EMBED_CACHE_TEST)

# -------------------------------
# Classifier
# -------------------------------
clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=1
)

print("\n🚀 Training classifier...")
clf.fit(X_train_emb, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = clf.predict(X_test_emb)
f1 = f1_score(y_test, y_pred)

print("\n📊 URL Model Evaluation:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malicious"]
)
disp.plot(cmap="Blues")
plt.title("URL Model Confusion Matrix")
plt.show()

print(f"\n🎯 Final F1 Score: {f1:.4f}")

# -------------------------------
# Save BEST model always (with F1 in name)
# -------------------------------
f1_tag = f"{f1:.4f}".replace(".", "_")

MODEL_PATH_CLASSIFIER = f"models/url_classifier_f1_{f1_tag}.pkl"
MODEL_PATH_EMBEDDER = f"models/url_embedder_f1_{f1_tag}.pkl"

joblib.dump(clf, MODEL_PATH_CLASSIFIER)
joblib.dump(embedder, MODEL_PATH_EMBEDDER)

print("\n💾 Model saved:")
print(f"  Classifier → {MODEL_PATH_CLASSIFIER}")
print(f"  Embedder   → {MODEL_PATH_EMBEDDER}")