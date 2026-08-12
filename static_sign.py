import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
from collections import deque
import time
import socketio

# ================= SOCKET SETUP =================
sio = socketio.Client()
sio.connect("http://localhost:5000")

# ================= LOAD MODEL =================
model = tf.keras.models.load_model("models/landmark_model.h5")

with open("models/label_map_static.json") as f:
    label_map = json.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

# ================= VARIABLES =================
pred_q = deque(maxlen=7)
sentence = ""
current_letter = ""
last_added_letter = ""
stable_counter = 0
last_prediction_time = time.time()

# ================= LANDMARK FUNCTION =================
def landmarks_to_vec(results):
    vec = np.zeros(21 * 3 * 2, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        hand_map = {}

        for idx, hland in enumerate(results.multi_hand_landmarks):
            label_h = results.multi_handedness[idx].classification[0].label
            coords = []
            for lm in hland.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            hand_map[label_h] = coords

        left = hand_map.get("Left", [0] * (21 * 3))
        right = hand_map.get("Right", [0] * (21 * 3))

        vec = np.array(left + right, dtype=np.float32)

    return vec


print("Static Sign Module Started...")

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    vec = landmarks_to_vec(results)

    # Prediction
    pred = model.predict(vec.reshape(1, -1), verbose=0)[0]
    idx = int(np.argmax(pred))
    conf = float(np.max(pred))

    pred_q.append((idx, conf))

    # Majority voting
    votes = {}
    for p, c in pred_q:
        votes[p] = votes.get(p, 0) + 1

    final_idx = max(votes, key=votes.get)
    final_conf = max(c for (i, c) in pred_q if i == final_idx)

    if final_conf > 0.6:
        detected_letter = inv_label_map[final_idx]

        if detected_letter == current_letter:
            stable_counter += 1
        else:
            current_letter = detected_letter
            stable_counter = 0

        if stable_counter > 12 and detected_letter != last_added_letter:
            sentence += detected_letter
            last_added_letter = detected_letter
            stable_counter = 0
            last_prediction_time = time.time()

            # 🔥 SEND TO FLASK SERVER
            sio.emit("send_text", {"text": sentence})

        display_text = f"{detected_letter} ({final_conf:.2f})"

    else:
        display_text = "..."

    # Auto space after pause
    if time.time() - last_prediction_time > 2:
        if len(sentence) > 0 and sentence[-1] != " ":
            sentence += " "
            sio.emit("send_text", {"text": sentence})
        last_prediction_time = time.time()

    # ================= DISPLAY =================
    cv2.putText(img, "Letter: " + display_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.putText(img, "Sentence: " + sentence,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2)

    cv2.imshow("Static Sign Detection", img)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == 32:
        sentence += " "
    elif key == 8:
        sentence = sentence[:-1]
    elif key == ord('c'):
        sentence = ""

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
sio.disconnect()