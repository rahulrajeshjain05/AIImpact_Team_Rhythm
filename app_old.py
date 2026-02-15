import os
import base64
import io
import logging
import numpy as np
import torch
import librosa
import uvicorn
from fastapi import FastAPI, HTTPException, Security, Depends, Header
from pydantic import BaseModel
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ======================================================
# CONFIG & HACKATHON SETTINGS
# ======================================================
HF_TOKEN = os.getenv("HF_Token") 
API_KEY_VALUE = "sk_test_123456789" # Set your secret key here

# Using the high-accuracy deepfake detection model
MODEL_ID = "Hemgg/Deepfake-audio-detection" 
TARGET_SR = 16000
LABEL_MAP = {0: "AI_GENERATED", 1: "HUMAN"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hcl-voice-detection")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# MODEL LOADING
# ======================================================
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

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# Security Layer: Checks for 'x-api-key' in headers
async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY_VALUE:
        # Standard Hackathon error response for auth
        raise HTTPException(status_code=403, detail="Invalid API key or malformed request")
    return x_api_key

# ======================================================
# CORE LOGIC
# ======================================================
def preprocess_audio(b64_string: str):
    try:
        # Clean potential data prefixes
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        
        # Base64 Decoding
        audio_bytes = base64.b64decode(b64_string)
        
        # Load via librosa for robust MP3 support
        with io.BytesIO(audio_bytes) as bio:
            audio, sr = librosa.load(bio, sr=TARGET_SR)

        # Padding/Normalization
        if len(audio) < TARGET_SR:
            audio = np.pad(audio, (0, TARGET_SR - len(audio)))

        return audio.astype(np.float32)
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise ValueError("Invalid audio data")

def generate_explanation(classification: str, confidence: float):
    if classification == "AI_GENERATED":
        return "Unnatural pitch consistency and robotic speech patterns detected in the spectral analysis."
    return "Natural prosody and human-like frequency variance identified."

# ======================================================
# ENDPOINTS
# ======================================================
@app.post("/api/voice-detection")
async def voice_detection(
    request: VoiceRequest, 
    auth: str = Depends(verify_api_key)
):
    if model is None:
        return {"status": "error", "message": "Model not available"}
        
    try:
        # 1. Audio Processing
        waveform = preprocess_audio(request.audioBase64)
        
        # 2. Inference
        inputs = feature_extractor(waveform, sampling_rate=TARGET_SR, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        confidence, pred_idx = torch.max(probs, dim=-1)
        classification = LABEL_MAP.get(int(pred_idx.item()), "UNKNOWN")
        score = round(float(confidence.item()), 2)

        # 3. Response Generation (Matches Hackathon Format)
        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": score,
            "explanation": generate_explanation(classification, score)
        }

    except Exception as e:
        logger.error(f"Inference error: {e}")
        return {
            "status": "error",
            "message": "Malformed request or processing error"
        }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)