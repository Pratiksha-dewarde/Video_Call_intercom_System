let handsModel = null;
let mpCamera = null;
let trackingActive = false;

const handCanvas = document.getElementById("handCanvas");
const handCtx = handCanvas.getContext("2d");

function initHandTracking() {
  if (userRole !== "deaf") return;
  if (trackingActive) return;

  if (typeof Hands === "undefined" || typeof Camera === "undefined") {
    console.error("MediaPipe Hands scripts did not load.");
    if (typeof appendMessage === "function") {
      appendMessage("Sign error: MediaPipe Hands did not load. Check internet access.", "sign");
    }
    return;
  }

  const videoEl = document.getElementById("localVideo");

  handsModel = new Hands({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
  });

  handsModel.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  handsModel.onResults(onHandsResults);

  mpCamera = new Camera(videoEl, {
    onFrame: async () => {
      if (handsModel && videoEl.readyState >= 2) {
        await handsModel.send({ image: videoEl });
      }
    },
    width: 640,
    height: 480
  });

  mpCamera.start();
  trackingActive = true;
  console.log("MediaPipe Hands tracking started");
}

function getHandLabel(handedness) {
  if (!handedness) return "Unknown";

  if (typeof handedness.label === "string") {
    return handedness.label;
  }

  if (
    handedness.classification &&
    handedness.classification[0] &&
    typeof handedness.classification[0].label === "string"
  ) {
    return handedness.classification[0].label;
  }

  if (handedness[0] && typeof handedness[0].label === "string") {
    return handedness[0].label;
  }

  return "Unknown";
}

function toLmArray(lmList) {
  if (!lmList) return [];
  return lmList.map((lm) => ({
    x: lm.x,
    y: lm.y,
    z: lm.z
  }));
}

function onHandsResults(results) {
  handCanvas.width = handCanvas.offsetWidth || 640;
  handCanvas.height = handCanvas.offsetHeight || 480;
  handCtx.clearRect(0, 0, handCanvas.width, handCanvas.height);

  const signBadge = document.getElementById("signBadge");
  const landmarksList = results.multiHandLandmarks || [];
  const handednessList = results.multiHandedness || [];
  const hasHands = landmarksList.length > 0;

  signBadge.style.display = hasHands ? "block" : "none";

  const payloadHands = [];

  for (let i = 0; i < landmarksList.length; i += 1) {
    const landmarks = landmarksList[i];
    const label = getHandLabel(handednessList[i]);

    if (typeof drawConnectors === "function") {
      drawConnectors(handCtx, landmarks, HAND_CONNECTIONS, {
        color: "#00d4aa",
        lineWidth: 2
      });

      drawLandmarks(handCtx, landmarks, {
        color: "#ff4f6d",
        lineWidth: 1,
        radius: 3
      });
    }

    payloadHands.push({
      label: label,
      landmarks: toLmArray(landmarks)
    });
  }

  if (payloadHands.length > 0) {
    socket.emit("hand_landmarks", { hands: payloadHands });
  }
}

function stopHandTracking() {
  if (mpCamera) {
    mpCamera.stop();
    mpCamera = null;
  }

  handsModel = null;
  trackingActive = false;

  if (handCtx) {
    handCtx.clearRect(0, 0, handCanvas.width, handCanvas.height);
  }

  const signBadge = document.getElementById("signBadge");
  if (signBadge) {
    signBadge.style.display = "none";
  }

  console.log("MediaPipe Hands tracking stopped");
}
