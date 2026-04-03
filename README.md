# Multimodal-AI-for-Email-and-URL-Threat-Detection-Text-Visual-Analysis-


# 🛡️ Multimodal AI for Phishing Detection

A multimodal artificial intelligence system that detects phishing and malicious threats by combining **textual analysis (emails & URLs)** and **visual analysis (website screenshots)**.

---

## 📌 Overview

Phishing attacks are becoming increasingly sophisticated, combining deceptive text, malicious URLs, and visually convincing fake websites. Traditional single-modality detection systems struggle to handle such complex threats.

This project introduces a **multimodal AI framework** that integrates:

- 📧 Email content analysis  
- 🔗 URL-based detection  
- 🖼️ Screenshot-based visual inspection  

The system improves detection accuracy by leveraging complementary information across all modalities.

---

## 🚀 Features

- ✅ Detect phishing using **email text, URLs, and images**
- ✅ Works with **single or multiple inputs**
- ✅ Combines predictions using **multimodal fusion**
- ✅ Provides **confidence scores**
- ✅ Interactive **Streamlit web application**
- ✅ High performance with optimized ML/DL models

---

## 🧠 System Architecture

The system consists of three independent models:

### 1. Email Classification
- TF-IDF vectorization
- Linear SVM (LinearSVC)
- Detects phishing language patterns

### 2. URL Classification
- Character-level TF-IDF (n-grams)
- Logistic Regression
- Detects obfuscated and malicious URLs

### 3. Image-Based Detection
- MobileNetV2 (Transfer Learning)
- Detects visual phishing cues (fake login pages, cloned layouts)

### 🔗 Multimodal Fusion
- Combines outputs using **weighted averaging**
- Works even if some inputs are missing

---

## 📊 Model Performance

| Modality | Accuracy | Precision | Recall | F1-Score |
|----------|----------|-----------|--------|----------|
| Email    | 0.9932   | 0.9934    | 0.9937 | 0.9935   |
| URL      | 0.9979   | 0.9964    | 0.9945 | 0.9954   |
| Image    | 0.8308   | 0.7183    | 0.8516 | 0.7793   |

> Multimodal approach outperforms single models, especially for complex phishing attacks. :contentReference[oaicite:1]{index=1}

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Machine Learning:** scikit-learn  
- **Deep Learning:** TensorFlow / PyTorch  
- **NLP:** TF-IDF, NLTK  
- **Image Processing:** OpenCV, PIL  
- **Web App:** Streamlit  

---


