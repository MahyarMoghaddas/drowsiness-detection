import os
import io
import json
import random
import cv2
import  numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

# ============================
# CONFIG
# ============================
DATA_DIR = os.path.join("dataset", "dataset_new")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_PATH   = os.path.join("models", "eye_model.h5")
HISTORY_PATH = os.path.join("models", "eye_history.json")

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Class indices output: {'Open': 0, 'Closed': 1}
CLASSES = ["Open", "Closed"]
IMG_SIZE = (24, 24)

# Academic style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.titlesize": 14
})

# Reproducibility
random.seed(42)

# ============================
# HELPERS
# ============================
def list_images(folder):
    imgs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                imgs.append(os.path.join(root, f))
    return imgs

def read_bgr(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image cannot be read: {path}")
    return img

def bgr_to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def preprocess_gray_resize(bgr, size=IMG_SIZE):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size)
    return gray, resized

def preprocess_for_model(gray_24):
    x = gray_24.astype("float32") / 255.0
    x = np.expand_dims(x, axis=-1)  # (24,24,1)
    x = np.expand_dims(x, axis=0)   # (1,24,24,1)
    return x

def predict_with_conf(model, gray_24):
    x = preprocess_for_model(gray_24)
    probs = model.predict(x, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    conf = float(np.max(probs)) * 100.0
    return pred_id, conf, probs

def save_fig(fig, filename, dpi=300):
    out = os.path.join(OUT_DIR, filename)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("✅ Saved:", out)

def ensure_test_data_exists():
    all_test = list_images(TEST_DIR)
    if len(all_test) == 0:
        raise RuntimeError(f"No images found under: {TEST_DIR}")
    return all_test

# ============================
# FIG 1: Dataset Samples Grid (clean)
# ============================
def fig1_dataset_samples(n_per_class=4):
    rows = len(CLASSES)
    cols = n_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(3.2*cols, 3.2*rows))

    if rows == 1:
        axes = np.array([axes])

    for r, cls in enumerate(CLASSES):
        cls_dir = os.path.join(TEST_DIR, cls)
        imgs = list_images(cls_dir)
        picks = random.sample(imgs, k=min(n_per_class, len(imgs))) if imgs else []

        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(picks):
                ax.imshow(bgr_to_rgb(read_bgr(picks[c])))
            if c == 0:
                ax.set_ylabel(f"Sınıf: {cls}", fontsize=12)

    fig.suptitle("Fig.1 — Test Veri Setinden Örnek Görüntüler (Open / Closed)", y=1.02)
    save_fig(fig, "fig1_dataset_samples.png", dpi=250)

# ============================
# FIG 2: Preprocessing Pipeline (clean)
# ============================
def fig2_preprocessing_pipeline():
    all_test = ensure_test_data_exists()
    path = random.choice(all_test)

    bgr = read_bgr(path)
    gray, resized = preprocess_gray_resize(bgr, IMG_SIZE)
    resized_big = cv2.resize(resized, (240, 240), interpolation=cv2.INTER_NEAREST)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(bgr_to_rgb(bgr))
    axes[0].set_title("Orijinal")
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    axes[2].imshow(resized_big, cmap="gray")
    axes[2].set_title("Resize: 24×24 (Görselleştirme için büyütüldü)")
    axes[2].axis("off")

    fig.suptitle("Fig.2 — Ön İşleme: Orijinal → Grayscale → 24×24", y=1.03)
    save_fig(fig, "fig2_preprocessing_pipeline.png", dpi=300)

# ============================
# FIG 3: Prediction Examples Grid (clean titles)
# ============================
def fig3_prediction_examples(n=8):
    model = load_model(MODEL_PATH)
    all_test = ensure_test_data_exists()

    picks = random.sample(all_test, k=min(n, len(all_test)))
    cols = 4
    rows = int(np.ceil(len(picks) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3.6*cols, 3.3*rows))
    axes = np.array(axes).reshape(rows, cols)

    for i, p in enumerate(picks):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        ax.axis("off")

        bgr = read_bgr(p)
        _, resized = preprocess_gray_resize(bgr, IMG_SIZE)

        pred_id, conf, _ = predict_with_conf(model, resized)
        pred_name = CLASSES[pred_id]

        ax.imshow(bgr_to_rgb(bgr))
        ax.set_title(f"Tahmin: {pred_name} | Güven: {conf:.1f}%", fontsize=10)

    # Hide unused cells
    for j in range(len(picks), rows*cols):
        rr = j // cols
        cc = j % cols
        axes[rr, cc].axis("off")

    fig.suptitle("Fig.3 — Model Tahmin Örnekleri (Test Seti)", y=1.02)
    save_fig(fig, "fig3_prediction_examples.png", dpi=250)

# ============================
# FIG 4: Confusion Matrix (Count)
# ============================
def fig4_confusion_matrix_counts():
    model = load_model(MODEL_PATH)

    y_true, y_pred = [], []

    for true_cls in CLASSES:
        folder = os.path.join(TEST_DIR, true_cls)
        imgs = list_images(folder)
        true_id = CLASSES.index(true_cls)

        for p in imgs:
            bgr = read_bgr(p)
            _, resized = preprocess_gray_resize(bgr, IMG_SIZE)
            pred_id, _, _ = predict_with_conf(model, resized)
            y_true.append(true_id)
            y_pred.append(pred_id)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    im = ax.imshow(cm)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Tahmin (Predicted)")
    ax.set_ylabel("Gerçek (True)")
    ax.set_title("Fig.4 — Confusion Matrix (Count)")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=13, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, "fig4_confusion_matrix_counts.png", dpi=300)

# ============================
# FIG 5: Confusion Matrix (Normalized %)
# ============================
def fig5_confusion_matrix_percent():
    model = load_model(MODEL_PATH)

    y_true, y_pred = [], []

    for true_cls in CLASSES:
        folder = os.path.join(TEST_DIR, true_cls)
        imgs = list_images(folder)
        true_id = CLASSES.index(true_cls)

        for p in imgs:
            bgr = read_bgr(p)
            _, resized = preprocess_gray_resize(bgr, IMG_SIZE)
            pred_id, _, _ = predict_with_conf(model, resized)
            y_true.append(true_id)
            y_pred.append(pred_id)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_pct = (cm.astype(np.float32) / row_sum) * 100.0

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    im = ax.imshow(cm_pct)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Tahmin (Predicted)")
    ax.set_ylabel("Gerçek (True)")
    ax.set_title("Fig.5 — Confusion Matrix (Normalize - %)")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_pct[i, j]:.1f}%",
                    ha="center", va="center",
                    color="white" if cm_pct[i, j] > 50 else "black",
                    fontsize=13, fontweight="bold")

    supports = cm.sum(axis=1)
    ax.text(-0.45, 2.05, f"Support (n): Open={supports[0]}, Closed={supports[1]}",
            fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, "fig5_confusion_matrix_percent.png", dpi=300)

# ============================
# FIG 6: Training Curves (Loss + Accuracy) from history json
# ============================
def fig6_training_curves_from_history():
    if not os.path.exists(HISTORY_PATH):
        print("⚠️ eye_history.json bulunamadı. Önce train_eye.py çalıştır.")
        return

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        hist = json.load(f)

    loss = hist.get("loss", [])
    val_loss = hist.get("val_loss", [])
    acc = hist.get("accuracy", [])
    val_acc = hist.get("val_accuracy", [])

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    # Loss
    axes[0].plot(loss, label="Train Loss")
    axes[0].plot(val_loss, label="Val Loss")
    axes[0].set_title("Loss (Kayıp) Eğrisi")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    # Accuracy
    axes[1].plot(acc, label="Train Accuracy")
    axes[1].plot(val_acc, label="Val Accuracy")
    axes[1].set_title("Accuracy (Doğruluk) Eğrisi")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle("Fig.6 — Eğitim Süreci: Loss ve Accuracy", y=1.02)
    save_fig(fig, "fig6_training_curves_loss_acc.png", dpi=300)

# ============================
# FIG 7: Model Summary (as image) — very academic
# ============================
def fig7_model_summary():
    model = load_model(MODEL_PATH)

    # Capture model.summary() text
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    summary_text = buf.getvalue().strip()

    fig = plt.figure(figsize=(9, 6))
    plt.axis("off")
    plt.title("Fig.7 — CNN Model Summary", pad=12)

    # Write text in a monospaced-like style
    plt.text(
        0.01, 0.98, summary_text,
        va="top", ha="left",
        family="monospace",
        fontsize=10
    )

    save_fig(fig, "fig7_model_summary.png", dpi=250)

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    # Check required files
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH} (Önce train_eye.py çalıştır.)")

    # Generate figures
    fig1_dataset_samples(n_per_class=4)
    fig2_preprocessing_pipeline()
    fig3_prediction_examples(n=8)
    fig4_confusion_matrix_counts()
    fig5_confusion_matrix_percent()
    fig6_training_curves_from_history()
    fig7_model_summary()

    print("\n🎉 Tüm akademik figürler 'figures/' klasörüne kaydedildi.")
