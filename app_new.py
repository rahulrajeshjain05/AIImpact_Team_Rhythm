import os
import base64
import logging
import tempfile
import numpy as np
import torch
import uvicorn

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from pydub import AudioSegment

# ======================================================
# CONFIGURATION
# ======================================================

MODEL_ID = "Hemgg/Deepfake-audio-detection"
HF_TOKEN = os.getenv("HF_TOKEN", None)

API_KEY_VALUE = os.getenv("API_KEY", "sk_test_123456789")

TARGET_SR = 16000
MAX_AUDIO_SECONDS = 8
MAX_LEN = TARGET_SR * MAX_AUDIO_SECONDS

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

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
# STARTUP: LOAD MODEL ONCE
# ======================================================

@app.on_event("startup")
def load_model():
    global model, feature_extractor

    try:
        logger.info("Loading model...")

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            MODEL_ID, token=HF_TOKEN
        )

        model = AutoModelForAudioClassification.from_pretrained(
            MODEL_ID, token=HF_TOKEN
        ).to(DEVICE)

        model.eval()

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None

# ======================================================
# API KEY VALIDATION
# ======================================================

async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY_VALUE:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key or malformed request"
        )
    return x_api_key

# ======================================================
# AUDIO PREPROCESSING (ROBUST)
# ======================================================

def preprocess_audio(b64_string: str):

    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]

        audio_bytes = base64.b64decode(b64_string)

        # Write to temporary file (handles malformed MP3)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()

            audio = AudioSegment.from_file(tmp.name)

        # convert to mono + 16kHz
        audio = audio.set_channels(1).set_frame_rate(TARGET_SR)

        samples = np.array(audio.get_array_of_samples()).astype(np.float32)

        # normalize safely
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples /= max_val

        # duration control
        samples = samples[:MAX_LEN]
        samples = np.pad(samples, (0, max(0, MAX_LEN - len(samples))))

        return samples

    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        raise HTTPException(
            status_code=400,
            detail="Invalid audio data"
        )

# ======================================================
# ACOUSTIC ANOMALY DETECTOR (SECOND SIGNAL)
# ======================================================

def acoustic_anomaly_score(waveform):

    energy_variance = np.var(np.abs(waveform))
    signal_variance = np.var(waveform)

    score = 0.0

    # low variance often indicates synthetic speech
    if energy_variance < 0.003:
        score += 0.5

    if signal_variance < 0.01:
        score += 0.5

    return min(score, 1.0)

# ======================================================
# DYNAMIC EXPLANATION
# ======================================================

def generate_explanation(waveform, classification):

    energy_variance = np.var(np.abs(waveform))
    signal_variance = np.var(waveform)

    if classification == "AI_GENERATED":

        if energy_variance < 0.003:
            return "Very uniform energy distribution and smooth spectral structure indicate synthetic voice characteristics"

        return "Unnatural spectral consistency and low vocal variation detected"

    else:

        if energy_variance > 0.01:
            return "Natural vocal fluctuations and human prosody patterns detected"

        return "Human-like frequency variation observed"

# ======================================================
# MAIN ENDPOINT
# ======================================================

@app.post("/api/voice-detection")
async def voice_detection(
    request: VoiceRequest,
    auth: str = Depends(verify_api_key)
):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not available"
        )

    # -----------------------------
    # INPUT VALIDATION
    # -----------------------------

    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported language"
        )

    if request.audioFormat.lower() != "mp3":
        raise HTTPException(
            status_code=400,
            detail="Only mp3 format supported"
        )

    try:

        # -----------------------------
        # PREPROCESS AUDIO
        # -----------------------------

        waveform = preprocess_audio(request.audioBase64)

        # -----------------------------
        # MODEL INFERENCE
        # -----------------------------

        inputs = feature_extractor(
            waveform,
            sampling_rate=TARGET_SR,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        model_confidence, pred_idx = torch.max(probs, dim=-1)
        model_score = float(model_confidence.item())

        # correct label mapping
        model_prediction = model.config.id2label[pred_idx.item()]

        # -----------------------------
        # SECOND SIGNAL: ACOUSTIC CHECK
        # -----------------------------

        anomaly_score = acoustic_anomaly_score(waveform)

        # ensemble scoring
        final_score = 0.8 * model_score + 0.2 * anomaly_score

        classification = (
            "AI_GENERATED" if final_score > 0.5 else "HUMAN"
        )

        confidence = round(float(final_score), 3)

        # -----------------------------
        # EXPLANATION
        # -----------------------------

        explanation = generate_explanation(waveform, classification)

        # -----------------------------
        # RESPONSE
        # -----------------------------

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
        raise HTTPException(
            status_code=400,
            detail="Malformed request or processing error"
        )

# ======================================================
# RUN SERVER
# ======================================================

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
