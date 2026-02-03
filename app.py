# ======================================================
# HCL AI VOICE DETECTION API
# Hugging Face Spaces (FastAPI)
# ======================================================

import base64
import io
import logging
import librosa
import torch

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ======================================================
# CONFIGURATION
# ======================================================
API_KEY_NAME = "access_token"
API_KEY_VALUE = "HCL_SECURE_KEY_2026"

MODEL_ID = "facebook/wav2vec2-base-960h"
TARGET_SR = 16000

# ======================================================
# LOGGING
# ======================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-detection")

# ======================================================
# DEVICE & MODEL LOADING (RUNS ON STARTUP)
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_ID,
    num_labels=2
).to(DEVICE)

model.eval()
logger.info("Model loaded successfully")

# ======================================================
# FASTAPI APP
# ======================================================
app = FastAPI(
    title="HCL AI Voice Detection API",
    version="1.0.0"
)

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# SCHEMAS
# ======================================================
class AudioRequest(BaseModel):
    audio_base64: str


class PredictionResponse(BaseModel):
    classification: str
    confidence_score: float


# ======================================================
# SECURITY
# ======================================================
async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY_VALUE:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key


# ======================================================
# CORE LOGIC
# ======================================================
def decode_audio(b64_audio: str) -> bytes:
    try:
        return base64.b64decode(b64_audio.split(",")[-1])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")


def analyze_voice(audio_bytes: bytes) -> tuple[str, float]:
    audio, _ = librosa.load(
        io.BytesIO(audio_bytes),
        sr=TARGET_SR,
        mono=True
    )

    inputs = feature_extractor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt"
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    confidence, prediction = torch.max(probs, dim=-1)
    label = "AI_GENERATED" if prediction.item() == 1 else "HUMAN"

    return label, round(confidence.item(), 4)


# ======================================================
# ENDPOINTS
# ======================================================
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(
    request: AudioRequest,
    _: str = Depends(verify_api_key)
):
    audio_bytes = decode_audio(request.audio_base64)
    label, score = analyze_voice(audio_bytes)

    return {
        "classification": label,
        "confidence_score": score
    }
