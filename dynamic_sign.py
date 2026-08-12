import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque, Counter
import socketio

# ================= SOCKET =================
sio = socketio.Client()
sio.connect("http://localhost:5000")

# ================= LOAD MODEL =================
model = tf.keras.models.load_model("models/best_model_dynamic.h5")

# Load labels
labels = []
with open("models/labels_dynamic.txt", "r") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) > 1:
            labels.append(parts[1])

# ================= MEDIAPIPE =================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================= PARAMETERS =================
SEQUENCE_LENGTH = 60
BUFFER_SIZE = 90
CONFIDENCE_THRESH = 0.75
VOTE_WINDOW = 5
PREDICT_EVERY = 10

frame_buffer = deque(maxlen=BUFFER_SIZE)
vote_buffer = deque(maxlen=VOTE_WINDOW)

frame_count = 0
gesture = "Waiting..."
confidence_val = 0.0

# ================= NORMALIZATION =================
def normalize(landmarks):
    lm = landmarks.copy()

    ref_idx = 15 * 3
    ref = lm[ref_idx: ref_idx + 3].copy()
    if np.all(ref == 0):
        ref = lm[0:3].copy()

    for i in range(0, len(lm), 3):
        lm[i] -= ref[0]
        lm[i+1] -= ref[1]
        lm[i+2] -= ref[2]

    ls = np.array([lm[11*3], lm[11*3+1], lm[11*3+2]])
    rs = np.array([lm[12*3], lm[12*3+1], lm[12*3+2]])
    scale = np.linalg.norm(ls - rs)

    if scale > 1e-5:
        lm = lm / scale

    return lm

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

print("Dynamic Sign Module Started...")

# ================= MAIN LOOP =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Convert
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic.process(image_rgb)

    # Landmark vector (225)
    landmarks = np.zeros(225, dtype=np.float32)

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[i*3] = lm.x
            landmarks[i*3+1] = lm.y
            landmarks[i*3+2] = lm.z

    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            landmarks[99 + i*3] = lm.x
            landmarks[99 + i*3 + 1] = lm.y
            landmarks[99 + i*3 + 2] = lm.z

    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            landmarks[162 + i*3] = lm.x
            landmarks[162 + i*3 + 1] = lm.y
            landmarks[162 + i*3 + 2] = lm.z

    frame_buffer.append(normalize(landmarks))

    # ================= PREDICTION =================
    if len(frame_buffer) >= SEQUENCE_LENGTH and frame_count % PREDICT_EVERY == 0:

        buf = np.array(frame_buffer)
        indices = np.linspace(0, len(buf)-1, SEQUENCE_LENGTH).astype(int)
        seq = buf[indices]

        input_data = np.expand_dims(seq, axis=0).astype(np.float32)
        probs = model(input_data, training=False).numpy()[0]

        predicted_idx = int(np.argmax(probs))
        confidence_val = float(probs[predicted_idx])

        if confidence_val >= CONFIDENCE_THRESH:
            vote_buffer.append(predicted_idx)

            if len(vote_buffer) == VOTE_WINDOW:
                most_common = Counter(vote_buffer).most_common(1)[0][0]
                gesture = labels[most_common]

                # 🔥 SEND TO SERVER
                sio.emit("send_text", {"text": gesture})

        else:
            gesture = "..."

    # ================= DISPLAY =================
    cv2.putText(frame, f"Sign: {gesture}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 100),
                2)

    cv2.putText(frame, f"Conf: {confidence_val:.2f}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2)

    cv2.imshow("Dynamic Sign Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == ord('c'):
        frame_buffer.clear()
        vote_buffer.clear()
        gesture = "Cleared"
        confidence_val = 0.0

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
holistic.close()
sio.disconnect()