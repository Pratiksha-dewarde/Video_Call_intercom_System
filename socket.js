const socket = io({ transports: ["websocket"] });

let userRole = null;
let userLang = "en";
let micActive = false;
let speechRecognizer = null;
let restartingSpeech = false;

let lastMessageKey = "";
let lastMessageAt = 0;
let lastSpeechText = "";
let lastSpeechAt = 0;

function selectRole(role) {
  userRole = role;

  document.getElementById("deafBtn").classList.toggle("selected", role === "deaf");
  document.getElementById("normalBtn").classList.toggle("selected", role === "normal");
  document.getElementById("enterBtn").disabled = false;

  document.getElementById("chatMode").textContent =
    role === "deaf" ? "Sign -> Text" : "Speech -> Text";
}

function selectLang(lang) {
  userLang = lang;
  document.getElementById("langEn").classList.toggle("active", lang === "en");
  document.getElementById("langHi").classList.toggle("active", lang === "hi");
  socket.emit("set_language", { lang: userLang });
}

function enterSession() {
  if (!userRole) return;

  socket.emit("set_role", { role: userRole });
  socket.emit("set_language", { lang: userLang });

  document.getElementById("setupPanel").style.display = "none";
  document.getElementById("session").style.display = "flex";

  if (userRole === "deaf") {
    document.getElementById("signControls").style.display = "flex";
    document.getElementById("localLabel").textContent = "You (Deaf)";
    document.getElementById("signBadge").style.display = "block";
  } else {
    document.getElementById("speechControls").style.display = "flex";
    document.getElementById("localLabel").textContent = "You (Normal)";
  }

  initCamera();
}

function toggleSpeech() {
  const btn = document.getElementById("micBtn");

  if (!micActive) {
    const started = startBrowserSpeechRecognition();
    if (!started) return;

    micActive = true;
    btn.classList.add("recording");
    btn.querySelector("span").textContent = "Stop Mic";
  } else {
    micActive = false;
    stopBrowserSpeechRecognition();
    btn.classList.remove("recording");
    btn.querySelector("span").textContent = "Start Mic";
  }
}

function startBrowserSpeechRecognition() {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    alert("Use Google Chrome or Microsoft Edge for accurate speech-to-text.");
    return false;
  }

  stopBrowserSpeechRecognition();

  speechRecognizer = new SpeechRecognition();
  speechRecognizer.lang = userLang === "hi" ? "hi-IN" : "en-IN";
  speechRecognizer.continuous = true;
  speechRecognizer.interimResults = false;
  speechRecognizer.maxAlternatives = 1;

  speechRecognizer.onresult = (event) => {
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const result = event.results[i];
      if (!result.isFinal) continue;

      const text = result[0].transcript.trim();
      sendSpeechText(text);
    }
  };

  speechRecognizer.onerror = (event) => {
    console.error("Speech recognition error:", event.error);

    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      appendMessage("Speech error: microphone permission was denied.", "speech");
      micActive = false;
      updateMicButton(false);
      return;
    }

    if (event.error === "network") {
      appendMessage("Speech error: browser speech recognition needs internet.", "speech");
    }
  };

  speechRecognizer.onend = () => {
    if (micActive && !restartingSpeech) {
      restartingSpeech = true;
      setTimeout(() => {
        restartingSpeech = false;
        if (micActive && speechRecognizer) {
          try {
            speechRecognizer.start();
          } catch (err) {
            console.warn("Speech restart failed:", err);
          }
        }
      }, 500);
    }
  };

  try {
    speechRecognizer.start();
    return true;
  } catch (err) {
    console.error("Speech start failed:", err);
    appendMessage("Speech error: could not start browser speech recognition.", "speech");
    return false;
  }
}

function stopBrowserSpeechRecognition() {
  if (!speechRecognizer) return;

  try {
    speechRecognizer.onend = null;
    speechRecognizer.stop();
  } catch (err) {
    console.warn("Speech stop ignored:", err);
  }

  speechRecognizer = null;
}

function sendSpeechText(text) {
  if (!text) return;

  const normalized = text.toLowerCase();
  const now = Date.now();

  if (normalized === lastSpeechText && now - lastSpeechAt < 2500) {
    return;
  }

  lastSpeechText = normalized;
  lastSpeechAt = now;

  socket.emit("speech_text", {
    text: text,
    lang: userLang
  });
}

function updateMicButton(active) {
  const btn = document.getElementById("micBtn");
  if (!btn) return;

  btn.classList.toggle("recording", active);
  btn.querySelector("span").textContent = active ? "Stop Mic" : "Start Mic";
}

function appendMessage(text, type) {
  const chatBox = document.getElementById("chatBox");
  const empty = chatBox.querySelector(".chat-empty");
  if (empty) empty.remove();

  const div = document.createElement("div");
  div.className = "message " + (type === "sign" ? "msg-deaf" : "msg-normal");
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function clearChat() {
  const chatBox = document.getElementById("chatBox");
  chatBox.innerHTML =
    '<div class="chat-empty">Messages will appear here once the call starts...</div>';
  lastMessageKey = "";
  lastMessageAt = 0;
}

socket.on("connect", () => {
  const dot = document.querySelector(".pulse-dot");
  const text = document.getElementById("statusText");
  dot.classList.add("connected");
  text.textContent = "Online";
});

socket.on("disconnect", () => {
  const dot = document.querySelector(".pulse-dot");
  const text = document.getElementById("statusText");
  dot.classList.remove("connected");
  text.textContent = "Disconnected";
});

socket.on("server_info", (data) => {
  console.log("Server:", data.msg);
});

socket.on("ai_status", (data) => {
  console.log("AI status:", data);
});

socket.on("peer_left", () => {
  const remVideo = document.getElementById("remoteVideo");
  if (remVideo) remVideo.srcObject = null;

  document.getElementById("noRemote").style.display = "flex";
  document.getElementById("startCallBtn").style.display = "flex";
  document.getElementById("endCallBtn").style.display = "none";
  document.getElementById("statusText").textContent = "Peer left";
});

socket.on("result", (data) => {
  const { text, type } = data;
  if (!text) return;

  const key = `${type}:${text}`;
  const now = Date.now();
  if (key === lastMessageKey && now - lastMessageAt < 1500) return;

  lastMessageKey = key;
  lastMessageAt = now;
  appendMessage(text, type);
});
