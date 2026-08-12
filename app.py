import json
import os
from collections import Counter, deque
from pathlib import Path

import eventlet

eventlet.monkey_patch()

import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room


BASE_DIR = Path(__file__).resolve().parent
ROOM = "intercom_room"


def first_existing(paths):
    for path in paths:
        if path and Path(path).exists():
            return Path(path)
    return None


# Static alphabet model:
# Trained from MediaPipe Hands as 21 landmarks * 3 coordinates * 2 hands = 126 features.
STATIC_SIGN_MODEL_PATH = first_existing(
    [
        os.getenv("STATIC_SIGN_MODEL_PATH"),
        BASE_DIR / "models" / "landmark_model.h5",
        r"C:\Users\prati\Desktop\SignLanguage_Project\notebooks\landmark_model.h5",
    ]
)

STATIC_LABEL_MAP_PATH = first_existing(
    [
        os.getenv("STATIC_LABEL_MAP_PATH"),
        BASE_DIR / "models" / "label_map_static.json",
        BASE_DIR / "models" / "label_map.json",
        r"C:\Users\prati\Desktop\SignLanguage_Project\notebooks\ISL_dataset_split\label_map.json",
    ]
)

sign_model = None
inv_label_map = {}

try:
    import tensorflow as tf

    if STATIC_SIGN_MODEL_PATH is None:
        print("Static sign model missing: models/landmark_model.h5")
    else:
        print(f"Loading static sign model: {STATIC_SIGN_MODEL_PATH}")
        sign_model = tf.keras.models.load_model(str(STATIC_SIGN_MODEL_PATH))
        print("Static sign model loaded.")
except Exception as exc:
    print(f"Static sign model not loaded: {exc}")
    sign_model = None

try:
    if STATIC_LABEL_MAP_PATH is None:
        print("Static label map missing: models/label_map_static.json")
    else:
        with STATIC_LABEL_MAP_PATH.open("r", encoding="utf-8") as label_file:
            label_map = json.load(label_file)
        inv_label_map = {int(index): label for label, index in label_map.items()}
        print(f"Static labels loaded: {inv_label_map}")
except Exception as exc:
    print(f"Static labels not loaded: {exc}")
    inv_label_map = {}


STATIC_CONFIDENCE_THRESH = 0.78
STATIC_MARGIN_THRESH = 0.15
STATIC_VOTE_WINDOW = 9
STATIC_MIN_VOTES = 6
PREDICT_EVERY = 2
DUPLICATE_COOLDOWN_FRAMES = 35


class SessionState:
    def __init__(self):
        self.vote_buffer = deque(maxlen=STATIC_VOTE_WINDOW)
        self.frame_count = 0
        self.last_output = ""
        self.last_emit_frame = -DUPLICATE_COOLDOWN_FRAMES


session_states = {}


app = Flask(__name__)
app.config["SECRET_KEY"] = "intercom-secret-2024"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")


@app.route("/")
def index():
    return render_template("index.html")


try:
    from ai_modules.speech import (
        accept_audio,
        get_status as get_speech_status,
        set_language,
        start_speech,
        stop_speech,
    )

    SPEECH_OK = True
except Exception as exc:
    SPEECH_OK = False
    print(f"Speech module unavailable: {exc}")

    def start_speech(*args, **kwargs):
        return False

    def stop_speech(*args, **kwargs):
        return None

    def set_language(*args, **kwargs):
        return None

    def accept_audio(*args, **kwargs):
        return None

    def get_speech_status():
        return {"available": False, "error": "Speech module import failed."}


def ai_status():
    speech_status = get_speech_status() if SPEECH_OK else {"available": False}
    return {
        "sign": {
            "mode": "static_alphabet",
            "ready": sign_model is not None and bool(inv_label_map),
            "model_path": str(STATIC_SIGN_MODEL_PATH) if STATIC_SIGN_MODEL_PATH else None,
            "labels_path": str(STATIC_LABEL_MAP_PATH) if STATIC_LABEL_MAP_PATH else None,
            "label_count": len(inv_label_map),
            "input_features": 126,
        },
        "speech": speech_status,
    }


def hand_to_coords(hand):
    coords = []
    for landmark in hand.get("landmarks", [])[:21]:
        coords.extend(
            [
                float(landmark.get("x", 0.0)),
                float(landmark.get("y", 0.0)),
                float(landmark.get("z", 0.0)),
            ]
        )

    if len(coords) < 63:
        coords.extend([0.0] * (63 - len(coords)))
    return coords[:63]


def build_static_vector(left, right):
    return np.array(left + right, dtype=np.float32)


def static_candidate_vectors(hands):
    """
    The model was trained as Left(63) + Right(63). Live webcam handedness can be
    swapped because of selfie/mirror behavior, so test both possible slots.
    """
    zero = [0.0] * 63
    parsed = []

    for hand in hands:
        coords = hand_to_coords(hand)
        if coords == zero:
            continue
        parsed.append(
            {
                "label": str(hand.get("label", "")).lower(),
                "coords": coords,
            }
        )

    if not parsed:
        return []

    if len(parsed) == 1:
        coords = parsed[0]["coords"]
        return [
            build_static_vector(coords, zero),
            build_static_vector(zero, coords),
        ]

    left = None
    right = None
    for item in parsed[:2]:
        if item["label"] == "left":
            left = item["coords"]
        elif item["label"] == "right":
            right = item["coords"]

    if left is None:
        left = parsed[0]["coords"]
    if right is None:
        right = parsed[1]["coords"] if len(parsed) > 1 else zero

    return [
        build_static_vector(left, right),
        build_static_vector(right, left),
    ]


def predict_static_sign(hands):
    best = None
    candidates = static_candidate_vectors(hands)

    for vec in candidates:
        probs = sign_model.predict(vec.reshape(1, -1), verbose=0)[0]
        predicted_idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        sorted_probs = np.sort(probs)
        second_best = float(sorted_probs[-2]) if len(sorted_probs) > 1 else 0.0
        margin = confidence - second_best

        if best is None or confidence > best["confidence"]:
            best = {
                "idx": predicted_idx,
                "confidence": confidence,
                "margin": margin,
            }

    return best


@socketio.on("connect")
def on_connect():
    sid = request.sid
    session_states[sid] = SessionState()
    join_room(ROOM, sid=sid)
    print(f"Connected {sid}")
    emit("server_info", {"msg": "Connected", "sid": sid})
    emit("ai_status", ai_status())


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    session_states.pop(sid, None)
    stop_speech(sid)
    leave_room(ROOM, sid=sid)
    print(f"Disconnected {sid}")
    emit("peer_left", {}, room=ROOM, include_self=False)


@socketio.on("set_role")
def on_set_role(data):
    print(f"Role set: {request.sid} -> {data.get('role')}")
    emit("role_confirmed", {"role": data.get("role", "normal")})


@socketio.on("offer")
def on_offer(data):
    emit("offer", data, room=ROOM, include_self=False)


@socketio.on("answer")
def on_answer(data):
    emit("answer", data, room=ROOM, include_self=False)


@socketio.on("candidate")
def on_candidate(data):
    emit("candidate", data, room=ROOM, include_self=False)


@socketio.on("end_call")
def on_end_call():
    emit("peer_left", {}, room=ROOM, include_self=False)
    print(f"Call ended by {request.sid}")


@socketio.on("hand_landmarks")
def on_hand_landmarks(data):
    if sign_model is None or not inv_label_map:
        return

    sid = request.sid
    state = session_states.get(sid)
    if state is None:
        return

    hands = data.get("hands", [])
    if not hands:
        state.vote_buffer.clear()
        return

    try:
        state.frame_count += 1
        if state.frame_count % PREDICT_EVERY != 0:
            return

        prediction = predict_static_sign(hands)
        if prediction is None:
            return

        predicted_idx = prediction["idx"]
        confidence = prediction["confidence"]
        margin = prediction["margin"]

        if confidence < STATIC_CONFIDENCE_THRESH or margin < STATIC_MARGIN_THRESH:
            state.vote_buffer.clear()
            return

        state.vote_buffer.append((predicted_idx, confidence))
        if len(state.vote_buffer) < STATIC_VOTE_WINDOW:
            return

        vote_counts = Counter(idx for idx, _ in state.vote_buffer)
        best_idx, best_count = vote_counts.most_common(1)[0]
        if best_count < STATIC_MIN_VOTES:
            return

        state.vote_buffer.clear()
        sign_text = inv_label_map.get(best_idx)
        if not sign_text:
            return

        can_repeat = (
            state.frame_count - state.last_emit_frame >= DUPLICATE_COOLDOWN_FRAMES
        )
        if sign_text != state.last_output or can_repeat:
            state.last_output = sign_text
            state.last_emit_frame = state.frame_count
            print(
                f"Static sign detected: {sign_text} "
                f"({confidence * 100:.0f}%, margin {margin * 100:.0f}%)"
            )
            emit(
                "result",
                {"text": sign_text, "type": "sign", "confidence": confidence},
                room=ROOM,
            )

    except Exception as exc:
        print(f"Static landmark error: {exc}")


@socketio.on("start_speech")
def on_start_speech(data=None):
    lang = (data or {}).get("lang", "en")
    print(f"Start speech for {request.sid} ({lang})")
    ok = start_speech(socketio, request.sid, lang=lang, room=ROOM)
    emit("speech_state", {"running": bool(ok), "lang": lang})


@socketio.on("stop_speech")
def on_stop_speech():
    print(f"Stop speech for {request.sid}")
    stop_speech(request.sid)
    emit("speech_state", {"running": False})


@socketio.on("set_language")
def on_set_language(data):
    lang = data.get("lang", "en")
    print(f"Language for {request.sid}: {lang}")
    set_language(request.sid, lang)


@socketio.on("speech_text")
def on_speech_text(data):
    text = str(data.get("text", "")).strip()
    lang = data.get("lang", "en")

    if not text:
        return

    print(f"Browser speech [{lang}]: {text}")
    emit(
        "result",
        {
            "text": text,
            "type": "speech",
            "lang": lang,
        },
        room=ROOM,
    )


@socketio.on("speech_audio")
def on_speech_audio(data):
    accept_audio(request.sid, data)


if __name__ == "__main__":
    print("Intercom server -> http://0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
