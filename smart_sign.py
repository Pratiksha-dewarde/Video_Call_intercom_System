"""
ai_modules/smart_sign.py  —  Fixed Version

This module is for SERVER-SIDE sign detection (if you process video frames
on the backend instead of using MediaPipe in the browser).

Fixes vs original:
  1. All models wrapped in try/except  →  module importable without model files
  2. static_model uses 63-feature vector (21 lm × 3)  matching app.py
     (original used 126 which was inconsistent with the browser landmark sender)
  3. dynamic_model label loading is robust (handles missing file)
  4. process_frame returns a consistent (text, annotated_img) tuple always
  5. Thread-safe: no shared mutable state at module level for predictions
  6. Motion threshold is configurable
  7. opencv import guarded  (not needed for landmark-only path)

NOTE: In the current architecture the browser (MediaPipe JS) does landmark
detection and sends 21 landmarks to the server via socket. This module is
provided for an ALTERNATIVE architecture where raw video frames are sent
to the server and processed here with Python MediaPipe + TF.
"""
c
import numpy as np
from collections import deque, Counter

# ─── Optional heavy imports ───────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    print("⚠️  opencv-python not installed — motion detection unavailable")

try:
    import mediapipe as mp
    mp_hands    = mp.solutions.hands
    mp_holistic = mp.solutions.holistic
    mp_drawing  = mp.solutions.drawing_utils
    MP_OK = True
except ImportError:
    MP_OK = False
    print("⚠️  mediapipe not installed — smart_sign unavailable")

try:
    import tensorflow as tf
    TF_OK = True
except ImportError:
    TF_OK = False
    print("⚠️  tensorflow not installed — smart_sign unavailable")

import json

# ─── Models ───────────────────────────────────────
static_model  = None
dynamic_model = None
inv_label_map_static  = {}
labels_dynamic        = []

def load_models():
    global static_model, dynamic_model, inv_label_map_static, labels_dynamic

    if not TF_OK:
        return

    try:
        static_model = tf.keras.models.load_model("models/landmark_model.h5")
        with open("models/label_map_static.json") as f:
            label_map = json.load(f)
        inv_label_map_static = {v: k for k, v in label_map.items()}
        print("✅  Static sign model loaded")
    except Exception as e:
        print(f"⚠️  Static model not loaded: {e}")

    try:
        dynamic_model = tf.keras.models.load_model("models/best_model_dynamic.h5")
        with open("models/labels_dynamic.txt") as f:
            labels_dynamic = [
                line.strip().split(maxsplit=1)[1]
                for line in f if line.strip() and len(line.strip().split()) > 1
            ]
        print("✅  Dynamic sign model loaded")
    except Exception as e:
        print(f"⚠️  Dynamic model not loaded: {e}")


# ─── MediaPipe instances ───────────────────────────
_hands    = None
_holistic = None

def _get_hands():
    global _hands
    if _hands is None and MP_OK:
        _hands = mp_hands.Hands(
            min_detection_confidence=0.65,
            min_tracking_confidence=0.60,
            max_num_hands=2,
        )
    return _hands

def _get_holistic():
    global _holistic
    if _holistic is None and MP_OK:
        _holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _holistic


# ─── Feature extraction ───────────────────────────

def get_static_vector(hand_results) -> np.ndarray:
    """21 landmarks × 3 = 63 floats  (single hand)."""
    vec = np.zeros(63, dtype=np.float32)
    if hand_results.multi_hand_landmarks:
        for i, lm in enumerate(hand_results.multi_hand_landmarks[0].landmark):
            vec[i*3:i*3+3] = [lm.x, lm.y, lm.z]
    return vec


def get_dynamic_vector(holistic_results) -> np.ndarray:
    """pose(33×3) + left_hand(21×3) + right_hand(21×3) = 225 floats."""
    vec = np.zeros(225, dtype=np.float32)

    if holistic_results.pose_landmarks:
        for i, lm in enumerate(holistic_results.pose_landmarks.landmark):
            vec[i*3:i*3+3] = [lm.x, lm.y, lm.z]

    if holistic_results.left_hand_landmarks:
        base = 33 * 3   # = 99
        for i, lm in enumerate(holistic_results.left_hand_landmarks.landmark):
            vec[base + i*3 : base + i*3 + 3] = [lm.x, lm.y, lm.z]

    if holistic_results.right_hand_landmarks:
        base = 33 * 3 + 21 * 3   # = 162
        for i, lm in enumerate(holistic_results.right_hand_landmarks.landmark):
            vec[base + i*3 : base + i*3 + 3] = [lm.x, lm.y, lm.z]

    return vec


def detect_motion(current_frame, prev_frame, threshold: float = 5.0) -> str:
    """Returns 'static' or 'dynamic'."""
    if prev_frame is None or not CV2_OK:
        return "static"
    diff  = cv2.absdiff(current_frame, prev_frame)
    gray  = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    score = float(np.mean(gray))
    return "dynamic" if score >= threshold else "static"


# ─── Per-session state  (instantiate one SmartSignDetector per user) ──

class SmartSignDetector:
    """
    Stateful detector for one user session.
    Call  process_frame(bgr_frame)  to get (text_or_empty, annotated_frame).
    """

    def __init__(self, motion_threshold: float = 5.0, static_conf: float = 0.85,
                 dynamic_conf: float = 0.90):
        self.motion_threshold = motion_threshold
        self.static_conf      = static_conf
        self.dynamic_conf     = dynamic_conf

        self.frame_buffer = deque(maxlen=30)
        self.vote_buffer  = deque(maxlen=5)
        self.last_output  = ""
        self.prev_frame   = None

    def process_frame(self, frame: "np.ndarray") -> tuple[str, "np.ndarray"]:
        if not MP_OK or not CV2_OK:
            return "", frame

        img = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hand_results     = _get_hands().process(rgb)
        holistic_results = _get_holistic().process(rgb)

        # ── draw skeleton ──────────────────────────
        if hand_results.multi_hand_landmarks:
            for hl in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, hl, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 212, 170), thickness=2),
                    mp_drawing.DrawingSpec(color=(255, 79, 109), thickness=2),
                )

        # ── no hand detected ──────────────────────
        if not hand_results.multi_hand_landmarks:
            self.prev_frame = img
            return "", img

        # ── motion switch ─────────────────────────
        mode = detect_motion(img, self.prev_frame, self.motion_threshold)
        self.prev_frame = img

        text = ""

        if mode == "static" and static_model is not None:
            vec  = get_static_vector(hand_results)
            pred = static_model.predict(vec.reshape(1, -1), verbose=0)[0]
            idx  = int(np.argmax(pred))
            conf = float(np.max(pred))

            if conf >= self.static_conf:
                letter = inv_label_map_static.get(idx, "")
                if letter and letter != self.last_output:
                    self.last_output = letter
                    text = letter

        elif mode == "dynamic" and dynamic_model is not None:
            vec = get_dynamic_vector(holistic_results)
            self.frame_buffer.append(vec)

            if len(self.frame_buffer) == 30:
                seq   = np.array(self.frame_buffer, dtype=np.float32)
                probs = dynamic_model(np.expand_dims(seq, axis=0))[0]
                idx   = int(np.argmax(probs))
                conf  = float(np.max(probs))

                if conf >= self.dynamic_conf:
                    self.vote_buffer.append(idx)

                    if len(self.vote_buffer) == 5:
                        best = Counter(self.vote_buffer).most_common(1)[0][0]
                        self.vote_buffer.clear()

                        if best < len(labels_dynamic):
                            word = labels_dynamic[best]
                            if word and word != self.last_output:
                                self.last_output = word
                                text = word

        return text, img


# ─── Module-level convenience (single global detector) ────────────────

_detector: SmartSignDetector | None = None

def process_frame(frame):
    """Legacy API: module-level single detector."""
    global _detector
    if _detector is None:
        load_models()
        _detector = SmartSignDetector()
    return _detector.process_frame(frame)
