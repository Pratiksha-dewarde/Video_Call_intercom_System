"""
Browser microphone -> Vosk speech recognition.

This replaces server-side sounddevice capture. In a real video intercom the
speaker is usually on a browser/device, not beside the Flask server microphone,
so the browser sends 16 kHz PCM chunks through Socket.IO.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

try:
    from vosk import KaldiRecognizer, Model

    VOSK_OK = True
except Exception as exc:  # pragma: no cover - depends on local environment
    KaldiRecognizer = None
    Model = None
    VOSK_OK = False
    VOSK_ERROR = str(exc)
else:
    VOSK_ERROR = ""


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_CANDIDATES = {
    "en": [
        os.getenv("VOSK_EN_PATH"),
        BASE_DIR / "models" / "vosk-model-en-in-0.5",
        r"C:\Users\prati\Desktop\SignLanguage_Project\speech2text\vosk-model-en-in-0.5",
    ],
    "hi": [
        os.getenv("VOSK_HI_PATH"),
        BASE_DIR / "models" / "vosk-model-small-hi-0.22",
        r"C:\Users\prati\Desktop\SignLanguage_Project\speech2text\vosk-model-small-hi-0.22",
    ],
}

SAMPLE_RATE = 16000
_model_cache: dict[str, Model] = {}
_sessions: dict[str, "BrowserVoskSession"] = {}


def _model_path(lang: str) -> Path | None:
    for candidate in MODEL_CANDIDATES.get(lang, MODEL_CANDIDATES["en"]):
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def _load_model(lang: str) -> Model:
    if not VOSK_OK:
        raise RuntimeError(f"Vosk import failed: {VOSK_ERROR}")

    lang = lang if lang in MODEL_CANDIDATES else "en"
    if lang not in _model_cache:
        path = _model_path(lang)
        if path is None:
            raise FileNotFoundError(
                f"Vosk model for '{lang}' not found. Set VOSK_EN_PATH/VOSK_HI_PATH "
                "or place the model folder inside project models/."
            )
        print(f"Loading Vosk {lang} model: {path}")
        _model_cache[lang] = Model(str(path))
        print(f"Vosk {lang} model ready.")
    return _model_cache[lang]


def get_status():
    return {
        "available": VOSK_OK,
        "error": VOSK_ERROR,
        "sample_rate": SAMPLE_RATE,
        "models": {
            "en": str(_model_path("en")) if _model_path("en") else None,
            "hi": str(_model_path("hi")) if _model_path("hi") else None,
        },
    }


class BrowserVoskSession:
    def __init__(self, socketio, sid: str, room: str):
        self.socketio = socketio
        self.sid = sid
        self.room = room
        self.lang = "en"
        self.recognizer = None
        self.running = False
        self.lock = threading.Lock()
        self.last_text = ""
        self.last_emit_time = 0.0

    def _make_recognizer(self, lang: str):
        model = _load_model(lang)
        recognizer = KaldiRecognizer(model, SAMPLE_RATE)
        recognizer.SetWords(True)
        return recognizer

    def set_language(self, lang: str):
        if lang not in MODEL_CANDIDATES:
            lang = "en"
        with self.lock:
            self.lang = lang
            if self.running:
                self.recognizer = self._make_recognizer(lang)
        print(f"Speech language for {self.sid}: {lang}")

    def start(self, lang: str = "en") -> bool:
        if lang not in MODEL_CANDIDATES:
            lang = "en"
        try:
            with self.lock:
                self.lang = lang
                self.recognizer = self._make_recognizer(lang)
                self.running = True
                self.last_text = ""
                self.last_emit_time = 0.0
        except Exception as exc:
            msg = str(exc)
            print(f"Speech start failed for {self.sid}: {msg}")
            self.socketio.emit(
                "speech_error",
                {"message": msg},
                to=self.sid,
                namespace="/",
            )
            return False

        print(f"Browser speech started for {self.sid} ({lang}).")
        return True

    def stop(self):
        with self.lock:
            self.running = False
            self.recognizer = None
        print(f"Browser speech stopped for {self.sid}.")

    def accept_audio(self, payload):
        if isinstance(payload, dict):
            payload = payload.get("audio", b"")
        if isinstance(payload, bytearray):
            payload = bytes(payload)
        if not isinstance(payload, bytes) or not payload:
            return

        with self.lock:
            if not self.running or self.recognizer is None:
                return
            recognizer = self.recognizer
            lang = self.lang

            accepted = recognizer.AcceptWaveform(payload)
            if not accepted:
                return

            try:
                result = json.loads(recognizer.Result())
            except Exception:
                result = {}

        text = result.get("text", "").strip()
        if not text:
            return

        now = time.time()
        if text == self.last_text and now - self.last_emit_time < 2.0:
            return

        self.last_text = text
        self.last_emit_time = now
        print(f"[{lang}] {text}")
        self.socketio.emit(
            "result",
            {"text": text, "type": "speech", "lang": lang},
            room=self.room,
            namespace="/",
        )


def _session(socketio, sid: str, room: str = "intercom_room") -> BrowserVoskSession:
    if sid not in _sessions:
        _sessions[sid] = BrowserVoskSession(socketio, sid, room)
    return _sessions[sid]


def start_speech(socketio, sid: str, lang: str = "en", room: str = "intercom_room"):
    return _session(socketio, sid, room).start(lang)


def stop_speech(sid: str):
    session = _sessions.get(sid)
    if session:
        session.stop()


def set_language(sid: str, lang: str):
    session = _sessions.get(sid)
    if session:
        session.set_language(lang)


def accept_audio(sid: str, payload):
    session = _sessions.get(sid)
    if session:
        session.accept_audio(payload)
