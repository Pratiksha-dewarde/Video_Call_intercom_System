/* ═══════════════════════════════════════════════════
   webrtc.js  — Fixed Version

   Fixes vs original:
   1. initCamera() called from enterSession(), NOT on load
      → avoids camera request before user picked a role
   2. startCall / endCall toggle button visibility correctly
   3. Incoming offer auto-answers (callee side)
   4. ICE candidates queued until remoteDescription is set
   5. endCall emits  end_call  to server so peer gets notified
   6. Multiple STUN servers for better LAN/NAT traversal
═══════════════════════════════════════════════════ */

let localStream = null;
let peer        = null;
let icePending  = [];   // buffer ICE candidates until remote desc is ready

const RTC_CONFIG = {
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    { urls: "stun:stun1.l.google.com:19302" },
  ],
};

// ─── Camera ───────────────────────────────────────

async function initCamera() {
  try {
    localStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: true,
    });

    document.getElementById("localVideo").srcObject = localStream;

    console.log("📷 Camera ready");

    // hand tracking starts after camera is up (deaf user only)
    if (typeof initHandTracking === "function" && userRole === "deaf") {
      initHandTracking();
    }

  } catch (err) {
    console.error("Camera error:", err);
    alert("Camera / microphone access denied. Please allow and reload.");
  }
}

// ─── Create peer connection ────────────────────────

function createPeer() {
  const pc = new RTCPeerConnection(RTC_CONFIG);

  // add our tracks
  localStream.getTracks().forEach(t => pc.addTrack(t, localStream));

  // receive remote tracks
  pc.ontrack = (e) => {
    const remVideo = document.getElementById("remoteVideo");
    remVideo.srcObject = e.streams[0];
    document.getElementById("noRemote").style.display = "none";
    document.getElementById("statusText").textContent = "In Call";
  };

  // send ICE
  pc.onicecandidate = (e) => {
    if (e.candidate) socket.emit("candidate", e.candidate);
  };

  // connection state feedback
  pc.onconnectionstatechange = () => {
    console.log("WebRTC state:", pc.connectionState);
    if (pc.connectionState === "disconnected" || pc.connectionState === "failed") {
      endCall(false);   // false = don't re-emit end_call to server
    }
  };

  return pc;
}

// ─── Start Call (caller side) ─────────────────────

async function startCall() {
  if (!localStream) { alert("Camera not ready yet"); return; }
  if (peer)         { console.log("Call already active"); return; }

  peer = createPeer();

  const offer = await peer.createOffer();
  await peer.setLocalDescription(offer);
  socket.emit("offer", peer.localDescription);

  setCallUI(true);
}

// ─── Receive Offer (callee side) ──────────────────

socket.on("offer", async (offer) => {
  console.log("📨 Received offer");

  if (peer) endCall(false);   // clean up any stale peer

  peer = createPeer();

  await peer.setRemoteDescription(new RTCSessionDescription(offer));

  // drain pending ICE
  for (const c of icePending) {
    try { await peer.addIceCandidate(new RTCIceCandidate(c)); } catch {}
  }
  icePending = [];

  const answer = await peer.createAnswer();
  await peer.setLocalDescription(answer);
  socket.emit("answer", answer);

  setCallUI(true);
});

// ─── Receive Answer ───────────────────────────────

socket.on("answer", async (answer) => {
  if (!peer) return;
  await peer.setRemoteDescription(new RTCSessionDescription(answer));

  // drain pending ICE
  for (const c of icePending) {
    try { await peer.addIceCandidate(new RTCIceCandidate(c)); } catch {}
  }
  icePending = [];
});

// ─── ICE candidate ────────────────────────────────

socket.on("candidate", async (candidate) => {
  if (peer && peer.remoteDescription) {
    try {
      await peer.addIceCandidate(new RTCIceCandidate(candidate));
    } catch (e) {
      console.warn("ICE add error:", e);
    }
  } else {
    // queue until remote description is set
    icePending.push(candidate);
  }
});

// ─── End Call ─────────────────────────────────────

function endCall(notifyServer = true) {
  console.log("📵 Ending call");

  if (peer) {
    peer.ontrack        = null;
    peer.onicecandidate = null;
    peer.close();
    peer = null;
  }

  icePending = [];

  const remVideo = document.getElementById("remoteVideo");
  if (remVideo) remVideo.srcObject = null;
  document.getElementById("noRemote").style.display = "flex";

  if (notifyServer) socket.emit("end_call");

  setCallUI(false);
}

// ─── UI helpers ───────────────────────────────────

function setCallUI(inCall) {
  document.getElementById("startCallBtn").style.display = inCall ? "none"  : "flex";
  document.getElementById("endCallBtn").style.display   = inCall ? "flex"  : "none";
  document.getElementById("statusText").textContent     = inCall ? "In Call" : "Online";
}

// clean up on tab close
window.addEventListener("beforeunload", () => endCall(true));
