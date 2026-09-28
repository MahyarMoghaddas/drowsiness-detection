import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------------------------------
# VERİ SETİ YOLLARI
# -------------------------------------------------
BASE_DIR = os.path.join("dataset", "dataset_new")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")

# Sadece göz durumu sınıfları (yawn/no_yawn kullanılmıyor)
EYE_CLASSES = ["Open", "Closed"]

# -------------------------------------------------
# ÇIKTI KLASÖRÜ VE DOSYALARI
# -------------------------------------------------
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(OUT_DIR, "eye_model.h5")
HISTORY_PATH = os.path.join(OUT_DIR, "eye_history.json")
PLOT_PATH    = os.path.join(OUT_DIR, "eye_training_curves_loss_acc.png")

# -------------------------------------------------
# EĞİTİM AYARLARI
# -------------------------------------------------
IMG_SIZE = (24, 24)
BATCH_SIZE = 32
EPOCHS = 15

# -------------------------------------------------
# DATA GENERATOR (VERİ ARTIRMA)
# -------------------------------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.10,
    brightness_range=(0.8, 1.2)
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    classes=EYE_CLASSES,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    classes=EYE_CLASSES,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Class indices:", train_data.class_indices)

# -------------------------------------------------
# CNN MODELİ
# -------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),

    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------------------------
# CALLBACKS
# -------------------------------------------------
callbacks = [
    EarlyStopping(
        monitor="val_loss",          # akademik olarak loss izlemek daha mantıklı
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# -------------------------------------------------
# MODEL EĞİTİMİ
# -------------------------------------------------
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data,
    callbacks=callbacks
)

# -------------------------------------------------
# HISTORY KAYDETME
# -------------------------------------------------
with open(HISTORY_PATH, "w", encoding="utf-8") as f:
    json.dump(history.history, f, ensure_ascii=False, indent=2)

print(f"✅ History kaydedildi: {HISTORY_PATH}")

# -------------------------------------------------
# EN İYİ MODEL ZATEN ModelCheckpoint ile kaydedildi
# İstersen ayrıca garanti için tekrar kaydet:
# model.save(MODEL_PATH)
# -------------------------------------------------
print(f"✅ En iyi model kaydedildi: {MODEL_PATH}")

# -------------------------------------------------
# EĞİTİM GRAFİĞİ (LOSS + ACC)
# -------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# LOSS
axes[0].plot(history.history.get("loss", []), label="Train Loss")
axes[0].plot(history.history.get("val_loss", []), label="Val Loss")
axes[0].set_title("Loss (Kayıp) Eğrisi")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)
axes[0].legend()

# ACCURACY
axes[1].plot(history.history.get("accuracy", []), label="Train Accuracy")
axes[1].plot(history.history.get("val_accuracy", []), label="Val Accuracy")
axes[1].set_title("Accuracy (Doğruluk) Eğrisi")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300)
plt.close()

print(f"✅ Eğitim grafiği kaydedildi: {PLOT_PATH}")

# -------------------------------------------------
# TEST SETİ DEĞERLENDİRME (Sunum için güzel)
# -------------------------------------------------
test_loss, test_acc = model.evaluate(test_data, verbose=0)
print(f"\n📌 Test Sonucu -> Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")
