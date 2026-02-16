import os
import base64
import logging
import tempfile
import subprocess
import numpy as np
import torch
import uvicorn
import soundfile as sf

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ======================================================
# CONFIGURATION
# ======================================================

MODEL_ID = "Hemgg/Deepfake-audio-detection"
HF_TOKEN = os.getenv("HF_TOKEN", None)

API_KEY_VALUE = os.getenv("API_KEY", "sk_RHYTHM_123456730")

TARGET_SR = 16000
MAX_AUDIO_SECONDS = 22
MAX_LEN = TARGET_SR * MAX_AUDIO_SECONDS

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

MODEL_TO_API_LABEL = {
    "HumanVoice": "HUMAN",
    "AIVoice": "AI_GENERATED",
    "human": "HUMAN",
    "ai": "AI_GENERATED",
    "REAL": "HUMAN",
    "FAKE": "AI_GENERATED"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-detection")

# ======================================================
# FASTAPI INIT
# ======================================================

app = FastAPI(title="AI Voice Detection API")

model = None
feature_extractor = None

# ======================================================
# REQUEST MODEL
# ======================================================

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# ======================================================
# STARTUP: LOAD MODEL
# ======================================================

@app.on_event("startup")
def load_model():
    global model, feature_extractor

    try:
        logger.info("Loading model...")

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN
        )

        model = AutoModelForAudioClassification.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN
        ).to(DEVICE)

        model.eval()
        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        model = None

# ======================================================
# API KEY VALIDATION
# ======================================================

async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY_VALUE:
        raise HTTPException(403, "Invalid API key or malformed request")
    return x_api_key

# ======================================================
# ROBUST AUDIO PREPROCESSING (FFMPEG BASED)
# ======================================================

def preprocess_audio(b64_string):

    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]

        audio_bytes = base64.b64decode(b64_string)

        if len(audio_bytes) < 1000:
            raise ValueError("Audio too small")

        with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in.flush()

            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_out:

                command = [
                    "ffmpeg",
                    "-y",
                    "-i", tmp_in.name,
                    "-ac", "1",
                    "-ar", str(TARGET_SR),
                    tmp_out.name
                ]

                subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )

                waveform, sr = sf.read(tmp_out.name)

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        waveform = waveform[:MAX_LEN]
        waveform = np.pad(waveform, (0, max(0, MAX_LEN - len(waveform))))

        return waveform.astype(np.float32)

    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        raise HTTPException(400, "Invalid audio data")

# ======================================================
# SAFE ACOUSTIC CHECK (ONLY CONFIDENCE ADJUSTMENT)
# ======================================================

def acoustic_confidence_adjustment(waveform, base_confidence):

    energy_var = np.var(np.abs(waveform))

    # very uniform energy → slightly increase AI confidence
    if energy_var < 0.002:
        return min(1.0, base_confidence + 0.05)

    # strong variation → slightly increase human confidence
    if energy_var > 0.02:
        return max(0.0, base_confidence - 0.05)

    return base_confidence

# ======================================================
# DYNAMIC EXPLANATION
# ======================================================

def generate_explanation(classification, confidence):

    if classification == "AI_GENERATED":
        if confidence > 0.9:
            return "Highly consistent spectral patterns indicate synthetic voice"
        return "Speech characteristics suggest AI-generated audio"

    else:
        if confidence > 0.9:
            return "Natural vocal variation and human prosody detected"
        return "Speech characteristics consistent with human voice"

# ======================================================
# MAIN ENDPOINT
# ======================================================

@app.post("/api/voice-detection")
async def voice_detection(
    request: VoiceRequest,
    auth: str = Depends(verify_api_key)
):

    if model is None:
        raise HTTPException(500, "Model not available")

    # ---------------- INPUT VALIDATION ----------------

    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, "Unsupported language")

    if request.audioFormat.lower() != "mp3":
        raise HTTPException(400, "Only mp3 format supported")

    try:

        # ---------------- PREPROCESS ----------------

        waveform = preprocess_audio(request.audioBase64)

        # ---------------- MODEL INFERENCE ----------------

        inputs = feature_extractor(
            waveform,
            sampling_rate=TARGET_SR,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        confidence, pred_idx = torch.max(probs, dim=-1)
        confidence = float(confidence.item())

        #classification = model.config.id2label[pred_idx.item()]
        model_prediction_raw = model.config.id2label[pred_idx.item()]

        classification = MODEL_TO_API_LABEL.get(
            model_prediction_raw,
            "AI_GENERATED" if "ai" in model_prediction_raw.lower() else "HUMAN"
        )


        # ---------------- SAFE CONFIDENCE ADJUSTMENT ----------------

        confidence = acoustic_confidence_adjustment(waveform, confidence)

        confidence = round(confidence, 3)

        # ---------------- EXPLANATION ----------------

        explanation = generate_explanation(classification, confidence)

        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": explanation
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(400, "Malformed request or processing error")

# ======================================================
# HEALTH CHECK
# ======================================================

@app.get("/")
def health():
    return {"status": "API running"}

# ======================================================
# RUN SERVER
# ======================================================

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
