import os

# =========================
# FORCE CPU ONLY (NO GPU)
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    accuracy_score
)

# =====================
# BASIC CONFIG
# =====================
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10
DATASET_PATH = "data/phish-iris"
MODEL_SAVE_PATH = "models/image_phishing_binary_cpu.keras"
THRESHOLD_SAVE_PATH = "models/image_model_threshold.txt"

os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("reports", exist_ok=True)


print("Using CPU only")
print("Available devices:", tf.config.list_physical_devices())

# =====================
# DATA GENERATORS
# =====================
train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.1
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = test_gen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Class indices:", train_data.class_indices)

# =====================
# MODEL (MobileNetV2)
# =====================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================
# TRAIN
# =====================
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)

# =====================
# TRAINING CURVES
# =====================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("reports/figures/image_training_curves.png", dpi=300)
plt.close()

# =====================
# PREDICTION
# =====================
print("\n🔎 Evaluating model...")

y_true = test_data.classes
y_probs = model.predict(test_data).ravel()

# =====================
# ROC – AUC
# =====================
fpr, tpr, roc_thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Image Phishing Model")
plt.legend(loc="lower right")
plt.grid(True)

plt.savefig("reports/figures/image_roc_curve.png", dpi=300)
plt.close()

print(f"ROC–AUC Score: {roc_auc:.4f}")

# =====================
# THRESHOLD TUNING
# =====================
thresholds = np.arange(0.30, 0.81, 0.05)
best_f1 = 0
best_threshold = 0.5
f1_scores = []

print("\n🔧 Threshold tuning (F1-score):")
for t in thresholds:
    y_pred_t = (y_probs >= t).astype(int)
    f1_t = f1_score(y_true, y_pred_t)
    f1_scores.append(f1_t)

    print(f"Threshold {t:.2f} → F1: {f1_t:.4f}")

    if f1_t > best_f1:
        best_f1 = f1_t
        best_threshold = t

plt.figure(figsize=(7, 5))
plt.plot(thresholds, f1_scores, marker="o")
plt.axvline(best_threshold, linestyle="--", label=f"Best Threshold = {best_threshold:.2f}")
plt.title("F1-score vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("F1-score")
plt.legend()
plt.grid(True)

plt.savefig("reports/figures/f1_vs_threshold.png", dpi=300)
plt.close()

# =====================
# FINAL EVALUATION
# =====================
y_pred_best = (y_probs >= best_threshold).astype(int)

accuracy = accuracy_score(y_true, y_pred_best)
f1 = f1_score(y_true, y_pred_best)
precision = precision_score(y_true, y_pred_best)
recall = recall_score(y_true, y_pred_best)
cm = confusion_matrix(y_true, y_pred_best)

print("\n📊 Confusion Matrix:")
print(cm)

print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred_best, target_names=["Legit", "Phishing"]))

print(f"Accuracy  : {accuracy:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")

# =====================
# SAVE CLASSIFICATION REPORT
# =====================
report_text = classification_report(
    y_true,
    y_pred_best,
    target_names=["Legit", "Phishing"]
)

with open("reports/classification_report_image.txt", "w") as f:
    f.write("Image-Based Phishing Detection Model\n")
    f.write("=================================\n\n")
    f.write(report_text)
    f.write("\n\n")
    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1-score  : {f1:.4f}\n")
    f.write(f"ROC–AUC   : {roc_auc:.4f}\n")

print("📄 Classification report saved to reports/classification_report_image.txt")


# =====================
# CONFUSION MATRIX PLOT
# =====================
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legit", "Phishing"],
    yticklabels=["Legit", "Phishing"]
)
plt.title("Confusion Matrix – Image Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

plt.savefig("reports/figures/image_confusion_matrix.png", dpi=300)
plt.close()

# =====================
# SAVE MODEL + THRESHOLD
# =====================
model.save(MODEL_SAVE_PATH)

with open(THRESHOLD_SAVE_PATH, "w") as f:
    f.write(str(best_threshold))

print("\n✅ Model saved:", MODEL_SAVE_PATH)
print("✅ Best threshold saved:", THRESHOLD_SAVE_PATH)