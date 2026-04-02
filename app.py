import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Multimodal Phishing Detection",
    layout="centered"
)

st.title("🔐 Multimodal Phishing Detection System")
st.write(
    "This system detects phishing using **image**, **URL**, and **email text**. "
    "You may provide **any combination** of inputs."
)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    image_model = load_model("models/image_phishing_binary_cpu.keras")
    url_model = joblib.load("models/url_char_clf_f1_nd_0.9954.pkl")
    url_vectorizer = joblib.load("models/url_char__nd_vectorizer.pkl")
    email_model = joblib.load("models/phishing_email_model_ccv.joblib")
    return image_model, url_model, email_model, url_vectorizer

image_model, url_model, email_model, url_vectorizer = load_models()

# =========================
# FUSION FUNCTION
# =========================
def fuse_predictions(preds):
    WEIGHTS = {
        "image": 0.25,
        "url": 0.35,
        "email": 0.4
    }

    active = {k: v for k, v in preds.items() if v is not None}

    if not active:
        return None

    total_weight = sum(WEIGHTS[k] for k in active)

    fused_prob = sum(
        (WEIGHTS[k] / total_weight) * active[k]
        for k in active
    )

    return fused_prob

# =========================
# INPUT SECTION
# =========================
st.header("📥 Input Data")

uploaded_image = st.file_uploader("Upload Website Screenshot (optional)", type=["jpg", "png", "jpeg"])
url_text = st.text_input("Enter URL (optional)")
email_text = st.text_area("Paste Email Content (optional)", height=150)

# =========================
# PREDICTION
# =========================
if st.button("🔍 Detect Phishing"):
    preds = {}

    # ---------- IMAGE ----------
    if uploaded_image is not None:
        img = Image.open(uploaded_image).convert("RGB")
        img = img.resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        img_prob = float(image_model.predict(img_array)[0][0])
        preds["image"] = img_prob

        st.write(f"🖼 **Image Model Confidence:** `{img_prob:.4f}`")

    # ---------- URL ----------
    if url_text.strip():
        try:
            url_vec = url_vectorizer.transform([url_text])
            url_prob = float(url_model.predict_proba(url_vec)[0][1])
            preds["url"] = url_prob

            st.write(f"🔗 **URL Model Confidence:** `{url_prob:.4f}`")
        except Exception as e:
            st.error(f"URL model error: {e}")

    # ---------- EMAIL ----------
    if email_text.strip():
        email_prob = float(email_model.predict_proba([email_text])[0][1])
        preds["email"] = email_prob

        st.write(f"📧 **Email Model Confidence:** `{email_prob:.4f}`")

    # ---------- FUSION ----------
    final_prob = fuse_predictions(preds)

    if final_prob is None:
        st.warning("⚠ Please provide at least one input.")
    else:
        st.divider()
        st.subheader("🧠 Final Decision")

        if final_prob >= 0.5:
            st.error(f"🚨 **Phishing Detected** (Confidence: {final_prob:.4f})")
        else:
            st.success(f"✅ **Legitimate** (Confidence: {1 - final_prob:.4f})")

# =========================
# FOOTER
# =========================
st.divider()
st.caption("CSIT985 – Multimodal AI Phishing Detection")