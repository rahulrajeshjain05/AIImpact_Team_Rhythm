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
# CONFIG & SECRETS
# ======================================================
API_KEY_NAME = "access_token"
API_KEY_VALUE = "HCL_SECURE_KEY_2026"

# Get your Hugging Face token from the Space's Secret settings
HF_TOKEN = os.getenv("HF_Token") 

MODEL_ID = "melba-t/wav2vec2-fake-speech-detection" 
TARGET_SR = 16000
LABEL_MAP = {0: "HUMAN", 1: "AI_GENERATED"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hcl-api")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# MODEL INITIALIZATION (WITH AUTH)
# ======================================================
try:
    logger.info(f"Loading private model {MODEL_ID}...")
    # Passing the token allows access to the private/restricted repo
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_ID, 
        token=HF_TOKEN
    )
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_ID, 
        token=HF_TOKEN
    ).to(DEVICE)
    model.eval()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    # Fallback to prevent app crash if token is missing
    model = None 

# ======================================================
# FASTAPI APP
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

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY_VALUE:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

def preprocess_audio(b64_string: str):
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        
        # Standardize padding
        missing_padding = len(b64_string) % 4
        if missing_padding:
            b64_string += "=" * (4 - missing_padding)

        audio_bytes = base64.b64decode(b64_string)
        
        # Load audio using librosa (requires ffmpeg in packages.txt)
        with io.BytesIO(audio_bytes) as bio:
            audio, sr = librosa.load(bio, sr=TARGET_SR)

        if len(audio) < TARGET_SR:
            audio = np.pad(audio, (0, TARGET_SR - len(audio)))

        return audio.astype(np.float32)
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise ValueError(f"Decoding failed: {str(e)}")

@app.get("/")
def home():
    return {"message": "HCL Voice Detection API Active. Visit /docs"}

@app.post("/predict")
async def predict(request: AudioRequest, _: str = Depends(verify_api_key)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check HF_Token.")
        
    try:
        waveform = preprocess_audio(request.audio_base64)
        
        inputs = feature_extractor(
            waveform, 
            sampling_rate=TARGET_SR, 
            return_tensors="pt"
        ).to(DEVICE)

        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        confidence, pred_idx = torch.max(probs, dim=-1)
        
        # Map prediction to required hackathon labels
        label = LABEL_MAP.get(int(pred_idx.item()), "UNKNOWN")

        return {
            "classification": label,
            "confidence_score": round(float(confidence.item()), 4)
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Inference Error")

if __name__ == "__main__":
    # Port 7860 is required for Hugging Face Spaces
    uvicorn.run("app:app", host="0.0.0.0", port=7860)