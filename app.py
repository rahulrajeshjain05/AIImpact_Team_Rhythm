# ======================================================
# HCL AI VOICE DETECTION API – HACKATHON SUBMISSION
# ======================================================

import base64
import io
import logging
import numpy as np
import torch
import soundfile as sf
import librosa

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ======================================================
# CONFIG & REQUIREMENTS MAPPING
# ======================================================
# The hackathon requires specific classification results
LABEL_MAP = {
    0: "HUMAN",
    1: "AI_GENERATED"
}

API_KEY_NAME = "access_token"
API_KEY_VALUE = "HCL_SECURE_KEY_2026"  # Ensure this matches your submission docs

# Using a model fine-tuned for Deepfake/Synthetic Voice Detection
MODEL_ID = "melba-t/wav2vec2-fake-speech-detection" 
TARGET_SR = 16000

# ======================================================
# INITIALIZATION
# ======================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hcl-voice-safety")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading model to {DEVICE}...")

try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")

# ======================================================
# FASTAPI SETUP
# ======================================================
app = FastAPI(title="HCL AI Voice Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioRequest(BaseModel):
    audio_base64: str

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# ======================================================
# UTILITIES
# ======================================================
async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY_VALUE:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

def preprocess_audio(b64_string: str):
    """Decodes base64 MP3/WAV and converts to 16kHz Mono."""
    try:
        # Strip header if present (e.g., data:audio/mp3;base64,...)
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        
        audio_bytes = base64.b64decode(b64_string)
        
        # Use soundfile for reading. Note: For MP3, ensure 'audioread' or 'ffmpeg' is in the environment
        with io.BytesIO(audio_bytes) as bio:
            audio, sr = sf.read(bio)

        # Convert to Mono if Stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample to 16kHz
        if sr != TARGET_SR:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)

        # Normalization & Padding for stability
        audio = np.nan_to_num(audio)
        if len(audio) < TARGET_SR:
            audio = np.pad(audio, (0, TARGET_SR - len(audio)))

        return audio.astype(np.float32)
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        raise ValueError("Could not decode audio. Ensure it is a valid Base64 MP3/WAV.")

# ======================================================
# ENDPOINTS
# ======================================================
@app.get("/health")
def health():
    return {"status": "active", "device": DEVICE}

@app.post("/predict")
async def predict(request: AudioRequest, _: str = Depends(verify_api_key)):
    """
    Analyzes voice sample and classifies as AI_GENERATED or HUMAN.
    """
    try:
        # 1. Preprocess
        waveform = preprocess_audio(request.audio_base64)
        
        # 2. Inference
        inputs = feature_extractor(
            waveform, 
            sampling_rate=TARGET_SR, 
            return_tensors="pt", 
            padding=True
        ).to(DEVICE)

        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        # 3. Get results
        confidence, pred_idx = torch.max(probs, dim=-1)
        label = LABEL_MAP.get(int(pred_idx.item()), "UNKNOWN")

        # 4. Return structured JSON
        return {
            "classification": label,
            "confidence_score": round(float(confidence.item()), 4)
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal server error during analysis")