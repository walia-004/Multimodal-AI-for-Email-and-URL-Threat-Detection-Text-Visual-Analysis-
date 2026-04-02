import joblib

clf = joblib.load("models/url_char_clf_f1_nd_0.9954.pkl")
vectorizer = joblib.load("models/url_char__nd_vectorizer.pkl")

while True:
    url = input("\nEnter URL: ").strip()
    if not url:
        break

    X = vectorizer.transform([url])
    prob = clf.predict_proba(X)[0][1]

    label = "Malicious" if prob >= 0.5 else "Benign"
    print(f"\n🔎 Prediction: {label}")
    print(f"📊 Confidence: {prob:.4f}")