import os
import base64
import io
import logging
import numpy as np
import torch
import librosa
import uvicorn
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ======================================================
# CONFIG & HACKATHON SETTINGS
# ======================================================
# Use the Secret "HF_Token" if the model ever becomes restricted
HF_TOKEN = os.getenv("HF_Token") 
API_KEY_NAME = "access_token"
API_KEY_VALUE = "HCL_SECURE_KEY_2026"

# A stable, high-accuracy public model for synthetic voice detection
MODEL_ID = "Hemgg/Deepfake-audio-detection" 
TARGET_SR = 16000

# Mapping model output indices to required Hackathon strings
# Note: Verified against Hemgg model config (0: Fake/AI, 1: Real/Human)
LABEL_MAP = {0: "AI_GENERATED", 1: "HUMAN"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hcl-voice-safety")

# ======================================================
# MODEL LOADING
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading model {MODEL_ID} to {DEVICE}...")

try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID, token=HF_TOKEN).to(DEVICE)
    model.eval()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Critical Error: Failed to load model: {e}")
    model = None

# ======================================================
# API SETUP
# ======================================================
app = FastAPI(title="HCL AI Voice Detection API")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioRequest(BaseModel):
    audio_base64: str

# Security layer
async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY_VALUE:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# ======================================================
# CORE LOGIC
# ======================================================
def preprocess_audio(b64_string: str):
    """Processes base64 audio into a normalized 16kHz waveform."""
    try:
        # Strip potential data URL prefix
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        
        # Ensure correct padding for base64
        missing_padding = len(b64_string) % 4
        if missing_padding:
            b64_string += "=" * (4 - missing_padding)

        audio_bytes = base64.b64decode(b64_string)
        
        # Load audio using librosa (backed by ffmpeg for MP3 support)
        with io.BytesIO(audio_bytes) as bio:
            audio, sr = librosa.load(bio, sr=TARGET_SR)

        # Padding/Stability: Ensure at least 1 second of audio
        if len(audio) < TARGET_SR:
            audio = np.pad(audio, (0, TARGET_SR - len(audio)))

        return audio.astype(np.float32)
    except Exception as e:
        logger.error(f"Audio Preprocessing Failed: {e}")
        raise ValueError("Decoding failed. Ensure valid Base64 MP3/WAV.")

@app.get("/")
def root():
    return {"status": "online", "model": MODEL_ID}

@app.post("/predict")
async def predict(request: AudioRequest, _: str = Depends(verify_api_key)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable.")
        
    try:
        # 1. Convert B64 to raw waveform
        waveform = preprocess_audio(request.audio_base64)
        
        # 2. Extract features and move to GPU/CPU
        inputs = feature_extractor(waveform, sampling_rate=TARGET_SR, return_tensors="pt").to(DEVICE)

        # 3. Model Inference (No Gradient Tracking)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        # 4. Map result to confidence and label
        confidence, pred_idx = torch.max(probs, dim=-1)
        label = LABEL_MAP.get(int(pred_idx.item()), "UNKNOWN")

        return {
            "classification": label,
            "confidence_score": round(float(confidence.item()), 4)
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Inference error occurred")
        raise HTTPException(status_code=500, detail="Internal server error.")

if __name__ == "__main__":
    # Standard port for Hugging Face Spaces
    uvicorn.run("app:app", host="0.0.0.0", port=7860)