import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

# =========================
# Create directories
# =========================
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# =========================
# Load data
# =========================
df = pd.read_csv("data/urldata.csv")

X = df["url"].astype(str)
y = df["result"].astype(int)

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# Vectorization
# =========================
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=5,
    max_features=50000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# Model
# =========================
clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=1
)

clf.fit(X_train_vec, y_train)

# =========================
# Predictions
# =========================
y_pred = clf.predict(X_test_vec)
y_prob = clf.predict_proba(X_test_vec)[:, 1]

f1 = f1_score(y_test, y_pred)

# =========================
# Save model & vectorizer
# =========================
joblib.dump(clf, "models/url_char_clf.pkl")
joblib.dump(vectorizer, "models/url_char_vectorizer.pkl")

# =========================
# Save classification report
# =========================
report = classification_report(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

with open("reports/classification_report_url.txt", "w") as f:
    f.write(report)

with open("reports/metrics_url.txt", "w") as f:
    f.write(f"Accuracy : {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall   : {rec:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
    f.write(f"Train samples: {len(X_train)}\n")
    f.write(f"Test samples : {len(X_test)}\n")

print("\n📊 Evaluation:")
print(report)
print(f"🎯 Accuracy : {acc:.4f}")
print(f"🎯 Precision: {prec:.4f}")
print(f"🎯 Recall   : {rec:.4f}")
print(f"🎯 F1 Score : {f1:.4f}")

# =========================
# Confusion Matrix Plot
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.xticks([0, 1])
plt.yticks([0, 1])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.savefig("plots/confusion_matrix_url.png")
plt.close()

# =========================
# ROC Curve
# =========================
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("plots/roc_curve_url.png")
plt.close()

# =========================
# Precision-Recall Curve
# =========================
precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig("plots/precision_recall_curve_url.png")
plt.close()

# =========================
# Feature Importance (Top 30)
# =========================
feature_names = np.array(vectorizer.get_feature_names_out())
coefs = clf.coef_[0]

top_positive = np.argsort(coefs)[-30:]
top_features = feature_names[top_positive]
top_weights = coefs[top_positive]

plt.figure(figsize=(8, 10))
plt.barh(top_features, top_weights)
plt.title("Top 30 Important Character N-Grams")
plt.xlabel("Coefficient Weight")
plt.tight_layout()
plt.savefig("plots/feature_importance_top30_url.png")
plt.close()

# =========================
# Class Distribution Plot
# =========================
plt.figure()
df["result"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig("plots/class_distribution_url.png")
plt.close()

print("\n✅ All models, reports, and plots saved successfully!")