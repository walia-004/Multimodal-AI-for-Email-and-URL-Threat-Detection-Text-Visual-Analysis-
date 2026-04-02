import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/malicious_phish.csv")
df["label_binary"] = df["type"].apply(lambda x: 0 if x == "benign" else 1)

X = df["url"].astype(str)
y = df["label_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=5,
    max_features=50000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=1
)

clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)
f1 = f1_score(y_test, y_pred)

print("\n📊 Evaluation:")
print(classification_report(y_test, y_pred))
print(f"\n🎯 F1 Score: {f1:.4f}")

joblib.dump(clf, f"models/url_char_clf_f1_{f1:.4f}.pkl")
joblib.dump(vectorizer, f"models/url_char_vectorizer.pkl")