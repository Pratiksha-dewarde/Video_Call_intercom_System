# SignBridge — IP Video Intercom for Deaf Communication

## Project Structure

```
intercom_fixed/
├── app.py                          ← Flask + SocketIO server (fixed)
├── requirements.txt
├── models/                         ← Place your model files here
│   ├── landmark_model.h5           (static sign classifier)
│   ├── label_map_static.json       (class index → letter)
│   ├── vosk-model-en-in-0.5/       (English speech model folder)
│   └── vosk-model-small-hi-0.22/   (Hindi speech model folder)
├── ai_modules/
│   ├── __init__.py
│   └── speech.py                   
├── templates/
│   └── index.html                  
└── static/
    ├── CSS/style.css               
    └── JS/
        ├── socket.js               
        ├── webrtc.js              
        └── hand_tracking.js       
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place model files in  models/  folder

# 3. Run the server
python app.py

# 4. Open two browser tabs (or two devices on same LAN)
#    http://localhost:5000
#    One tab → Deaf User
#    Other tab → Normal User
```

---


## MediaPipe Hand Skeleton (Deaf User)

The canvas overlay (`#handCanvas`) sits directly on top of `#localVideo` using `position:absolute; inset:0`. It draws:
- **Teal connectors** between hand joints
- **Red dots** at each of the 21 landmark points

This is only active for the **Deaf User** role.

---

## How the AI Pipeline Works

```
Deaf User camera
      │
      ▼
MediaPipe Hands (browser)
      │  21 landmarks × {x,y,z}
      ▼
socket.emit("landmarks", plain_array)
      │
      ▼
Flask server  →  TF Lite static model  →  letter/word
      │
      ▼
socket.emit("result", {text, type:"sign"})   (broadcast to room)
      │
      ▼
Normal User chat box
```

```
Normal User microphone (server mic via sounddevice)
      │
      ▼
Vosk KaldiRecognizer  →  transcript text
      │
      ▼
socket.emit("result", {text, type:"speech"})   (broadcast to room)
      │
      ▼
Deaf User chat box
```
