# 🎯 AI Generated Voice Detection API
### Detect AI-generated vs Human speech (Tamil, English, Hindi, Malayalam, Telugu)

A production-ready REST API that detects whether a voice recording is **AI generated** or **human spoken** using deep learning based audio classification.

This solution is designed for high accuracy, reliability, low latency, and secure API deployment.

---

# 🚀 Problem Statement

AI systems can generate extremely realistic human-like speech. This project detects whether a voice recording is:

• AI_GENERATED → Voice created using AI or synthetic systems  
• HUMAN → Voice spoken by a real human  

Supported languages:

- Tamil
- English
- Hindi
- Malayalam
- Telugu

The system accepts Base64 MP3 audio and returns classification via secure REST API.

---

# 🧠 Solution Approach

The system uses a pretrained deepfake audio detection model combined with robust audio preprocessing.

Detection Flow:

Audio Input  
↓  
Audio Normalization  
↓  
Deep Learning Audio Classifier (Wav2Vec2 based)  
↓  
Prediction Mapping  
↓  
Confidence Scoring  
↓  
Final Classification

---

# 🏗️ System Architecture

## High Level Architecture

Client Request  
│  
│ (Base64 MP3 + API Key)  
↓  
FastAPI Server  
│  
├── API Key Validation  
├── Input Validation  
├── Audio Preprocessing  
│  
↓  
Deep Learning Model (Wav2Vec2 Audio Classifier)  
↓  
Prediction Mapping Engine  
↓  
JSON Response

---

## Detailed Processing Pipeline

1. Request Received
   - Language validation
   - Audio format validation
   - API key verification

2. Audio Processing
   - Base64 decode
   - MP3 parsing
   - Convert to mono
   - Resample to 16kHz
   - Normalize waveform

3. Model Inference
   - Feature extraction
   - Deepfake detection model
   - Probability scoring

4. Output Generation
   - Label mapping
   - Confidence score
   - Explanation

5. JSON Response returned

---

# ⚙️ Technology Stack

API Framework → FastAPI  
Model Framework → PyTorch  
Audio Model → HuggingFace Transformers  
Audio Processing → Librosa / Pydub  
Deployment → HuggingFace Spaces / Docker  
Language → Python 3.10+

---

# 📦 Model Used

Hemgg/Deepfake-audio-detection

Model Capabilities:

- Deepfake speech detection
- Spectral pattern analysis
- Voice consistency analysis
- Human prosody detection
- Multi-language speech support

---

# 🔐 Security

The API uses header-based authentication.

Header format:

x-api-key: YOUR_SECRET_KEY

Requests without valid key are rejected.

---

# 📡 API Specification

## Endpoint

POST /api/voice-detection

---

## Request Format

{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_AUDIO_STRING"
}

---

## Success Response

{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Voice created using AI or synthetic systems"
}

OR

{
  "status": "success",
  "language": "English",
  "classification": "HUMAN",
  "confidenceScore": 0.87,
  "explanation": "Voice spoken by a real human"
}

---

## Error Response

{
  "status": "error",
  "message": "Invalid API key or malformed request"
}

---

# 🖥️ Local Setup Guide

## 1. Clone Repository

git clone <repo-url>  
cd voice-detection

---

## 2. Install Dependencies

pip install -r requirements.txt

---

## 3. Install FFmpeg (Required)

Ubuntu / HuggingFace Spaces:

apt-get update  
apt-get install -y ffmpeg

Mac:

brew install ffmpeg

Windows:

Download from https://ffmpeg.org/download.html

---

## 4. Run Server

python app.py

Server runs at:

http://localhost:7860

---

# 🧪 Testing API

## Using cURL

curl -X POST http://localhost:7860/api/voice-detection \
-H "Content-Type: application/json" \
-H "x-api-key: sk_test_123456789" \
-d @request.json

---

# 🧰 Google Colab Testing Workflow

1. Upload MP3 file
2. Convert to Base64
3. Send request to API
4. Verify classification

This ensures consistent evaluation.

---

# 📊 Evaluation Strategy

The system is designed to perform well on:

- AI generated speech (TTS, cloned voices)
- Natural human recordings
- Multiple languages
- Different recording qualities
- Short audio clips
- Noisy environments

Evaluation factors:

- Detection accuracy
- Language consistency
- Response reliability
- API performance
- Security validation

---

# ⚡ Performance Characteristics

- Model loads once at startup
- Low latency inference
- GPU support if available
- Handles malformed audio
- Scalable deployment

---

# ☁️ Deployment Architecture

User Request  
↓  
HuggingFace Space / Cloud Server  
↓  
FastAPI Application  
↓  
Model Inference  
↓  
JSON Response

Advantages:

- Zero server setup
- Auto scaling
- Public API endpoint
- Easy judge testing

---

# 🎯 Design Decisions

Why FastAPI:
- High performance async framework
- Production ready
- Easy deployment
- Low latency API responses

Why Wav2Vec2 Based Model:
- State of the art audio representation
- Robust speech features
- Good cross language generalization

Why 16kHz Sampling:
- Standard speech processing rate
- Model compatibility
- Reduced noise

---

# 📈 Future Improvements

- Ensemble multi-model detection
- Automatic language detection
- Confidence calibration
- Real-time streaming detection
- Speaker verification integration

---

# 👨‍💻 Author

AI Voice Detection Hackathon Submission.
