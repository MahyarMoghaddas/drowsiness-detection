import os
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from pygame import mixer

# =========================================================
# GENEL AYARLAR
# =========================================================

MODEL_PATH = os.path.join("models", "eye_model.h5")
ALARM_PATH = "alarm.wav"

FACE_CASCADE = os.path.join("haar cascade files", "haarcascade_frontalface_alt.xml")
LEFT_EYE_CASCADE = os.path.join("haar cascade files", "haarcascade_lefteye_2splits.xml")
RIGHT_EYE_CASCADE = os.path.join("haar cascade files", "haarcascade_righteye_2splits.xml")

IMG_SIZE = (24, 24)

# Alarm eşiği (frame sayısı)
SCORE_THRESHOLD = 10

# Class indices: {'Open': 0, 'Closed': 1}
OPEN_ID = 0
CLOSED_ID = 1

CAM_W, CAM_H = 1280, 720
# CAM_W, CAM_H = 640, 480

# Screenshot klasörü
ALERT_DIR = "alerts"
os.makedirs(ALERT_DIR, exist_ok=True)

# =========================================================
# MODEL + CASCADE
# =========================================================

model = load_model(MODEL_PATH)

face = cv2.CascadeClassifier(FACE_CASCADE)
leye = cv2.CascadeClassifier(LEFT_EYE_CASCADE)
reye = cv2.CascadeClassifier(RIGHT_EYE_CASCADE)

mixer.init()
sound = None
if os.path.exists(ALARM_PATH):
    sound = mixer.Sound(ALARM_PATH)
    sound.set_volume(1.0)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

font = cv2.FONT_HERSHEY_SIMPLEX

score = 0
thicc = 2
alarm_on = False
last_screenshot_time = 0.0

# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================

def preprocess_eye(eye_bgr):
    eye_gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
    eye_gray = cv2.resize(eye_gray, IMG_SIZE)
    eye_gray = eye_gray.astype("float32") / 255.0
    eye_gray = np.expand_dims(eye_gray, axis=-1)
    eye_gray = np.expand_dims(eye_gray, axis=0)
    return eye_gray


def predict_eye_with_confidence(eye_input):
    """
    Model tahmini:
    - class: Open / Closed
    - confidence: %
    """
    probs = model.predict(eye_input, verbose=0)[0]
    cls = int(np.argmax(probs))
    conf = float(np.max(probs)) * 100.0
    return cls, conf


def confidence_color(conf):
    if conf >= 80:
        return (0, 255, 0)      # Yeşil
    elif conf >= 60:
        return (0, 255, 255)    # Sarı
    else:
        return (0, 0, 255)      # Kırmızı


# =========================================================
# ANA DÖNGÜ
# =========================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    faces = face.detectMultiScale(gray, 1.1, 5, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 1)

    got_left = False
    got_right = False

    lpred, lconf = OPEN_ID, 0.0
    rpred, rconf = OPEN_ID, 0.0

    for (x, y, w, h) in right_eye:
        rpred, rconf = predict_eye_with_confidence(
            preprocess_eye(frame[y:y+h, x:x+w])
        )
        got_right = True
        break

    for (x, y, w, h) in left_eye:
        lpred, lconf = predict_eye_with_confidence(
            preprocess_eye(frame[y:y+h, x:x+w])
        )
        got_left = True
        break

    if not (got_left and got_right):
        cv2.putText(frame, "Eye not detected", (10, 30),
                    font, 0.8, (255, 255, 255), 2)
        cv2.imshow("Drowsiness", frame)
        if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
            break
        continue

    eye_conf = (lconf + rconf) / 2.0
    text_color = confidence_color(eye_conf)

    # ---------------- SCORE ----------------
    if lpred == CLOSED_ID and rpred == CLOSED_ID:
        score += 1
        status = "Closed"
    else:
        score = max(0, score - 1)
        status = "Open"

    # ---------------- UI ----------------
    cv2.putText(frame,
                f"Status: {status} ({eye_conf:.1f}%)",
                (10, 30),
                font, 0.8, text_color, 2)

    cv2.putText(frame,
                f"Score: {score}",
                (10, 65),
                font, 0.8, (255, 255, 255), 2)

    # ---------------- ALARM + SCREENSHOT ----------------
    if score > SCORE_THRESHOLD and not alarm_on:
        alarm_on = True

        if sound is not None:
            try:
                sound.play()
            except:
                pass

        now = time.time()
        if now - last_screenshot_time > 2.0:
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(ALERT_DIR, f"alert_{ts}.jpg"), frame)
            last_screenshot_time = now

    if score <= SCORE_THRESHOLD:
        alarm_on = False

    if score > SCORE_THRESHOLD:
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 4)

    cv2.imshow("Drowsiness", frame)

    if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
        break

cap.release()
cv2.destroyAllWindows()
