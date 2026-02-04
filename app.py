# ======================================================
# HCL AI VOICE DETECTION API – CRASH-PROOF VERSION
# ======================================================

import base64
import io
import logging
import numpy as np
import torch
import soundfile as sf
import librosa

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
# AUDIO DECODING (SAFE)
# ======================================================
def decode_audio(b64_audio: str):
    audio_bytes = base64.b64decode(b64_audio.split(",")[-1])
    audio, sr = sf.read(io.BytesIO(audio_bytes))

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != TARGET_SR:
        audio = librosa.resample(audio.astype(float), sr, TARGET_SR)

    audio = np.nan_to_num(audio)

    if len(audio) < TARGET_SR:
        audio = np.pad(audio, (0, TARGET_SR - len(audio)))

    return audio.astype(np.float32)

# ======================================================
# INFERENCE (CRASH-PROOF)
# ======================================================
def analyze_voice(audio):
    try:
        inputs = feature_extractor(
            audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        score, pred = torch.max(probs, dim=-1)

        return {
            "classification": "UNKNOWN",
            "confidence_score": round(score.item(), 4),
            "raw_label_index": int(pred.item())
        }

    except Exception as e:
        logger.exception("Model inference failed")
        return {
            "classification": "MODEL_ERROR",
            "confidence_score": 0.0,
            "error": str(e)
        }

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
    result = analyze_voice(audio)
    return result
