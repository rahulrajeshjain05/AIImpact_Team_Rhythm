# ======================================================
# HCL AI VOICE DETECTION API – HF SPACES (STABLE)
# ======================================================

import base64
import io
import logging
import torch
import soundfile as sf

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ======================================================
# CONFIG
# ======================================================
API_KEY_NAME = "access_token"
API_KEY_VALUE = "HCL_SECURE_KEY_2026"

# ✅ VERIFIED audio-classification model
MODEL_ID = "superb/wav2vec2-base-superb-ks"
TARGET_SR = 16000

# ======================================================
# LOGGING
# ======================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-detection")

# ======================================================
# DEVICE & MODEL
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

logger.info("Model loaded successfully")

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

# ======================================================
# SCHEMA
# ======================================================
class AudioRequest(BaseModel):
    audio_base64: str


# ======================================================
# SECURITY
# ======================================================
async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY_VALUE:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key


# ======================================================
# AUDIO + INFERENCE
# ======================================================
def decode_audio(b64_audio: str):
    try:
        audio_bytes = base64.b64decode(b64_audio.split(",")[-1])
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        if sr != TARGET_SR:
            raise ValueError("Audio must be 16kHz")
        return audio
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decode failed: {e}")


def analyze_voice(audio):
    inputs = feature_extractor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt"
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    confidence, pred = torch.max(probs, dim=-1)
    label = "AI_GENERATED" if pred.item() == 1 else "HUMAN"

    return label, round(confidence.item(), 4)


# ======================================================
# ENDPOINTS
# ======================================================
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/predict")
async def predict(
    request: AudioRequest,
    _: str = Depends(verify_api_key)
):
    audio = decode_audio(request.audio_base64)
    label, score = analyze_voice(audio)

    return {
        "classification": label,
        "confidence_score": score
    }
