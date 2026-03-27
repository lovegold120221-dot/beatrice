"""
FastAPI TTS Server using Coqui XTTS v2
Multilingual text-to-speech with streaming support
"""

import os

os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

import io
import base64
import torch
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from TTS.api import TTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global TTS instance
tts = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    logger.info("Initializing Coqui TTS (XTTS v2)...")
    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(
            f"TTS initialized on: {'cuda' if torch.cuda.is_available() else 'cpu'}"
        )
    except Exception as e:
        logger.error(f"Failed to initialize TTS: {e}")
        raise
    yield
    logger.info("Shutting down TTS server...")


app = FastAPI(
    title="Eburon TTS API",
    description="Coqui XTTS v2 Multilingual TTS Server",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = "en"
    speaker: Optional[str] = "Ana Florence"
    speaker_wav: Optional[str] = None


class StreamRequest(BaseModel):
    text: str
    language: Optional[str] = "en"
    speaker: Optional[str] = "Ana Florence"
    speaker_wav: Optional[str] = None


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Eburon TTS Server",
        "model": "xtts_v2",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.post("/tts")
async def synthesize_speech(request: TTSRequest):
    if not tts:
        raise HTTPException(status_code=503, detail="TTS not initialized")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        logger.info(
            f"Synthesizing: {request.text[:50]}... (lang={request.language}, speaker={request.speaker})"
        )

        import numpy as np

        wav = tts.tts(
            text=request.text,
            language=request.language,
            speaker=request.speaker,
            speaker_wav=request.speaker_wav if request.speaker_wav else None,
        )

        if isinstance(wav, list):
            wav = np.array(wav, dtype=np.float32)

        buffer = io.BytesIO()
        import scipy.io.wavfile as wavfile

        wavfile.write(buffer, rate=24000, data=wav)
        buffer.seek(0)

        audio_bytes = buffer.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()

        return {"audio": audio_b64, "sample_rate": 24000, "format": "wav"}
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/stream")
async def stream_speech(request: StreamRequest):
    if not tts:
        raise HTTPException(status_code=503, detail="TTS not initialized")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    def generate():
        try:
            logger.info(f"Streaming: {request.text[:50]}... (lang={request.language})")

            wav = tts.tts(
                text=request.text,
                language=request.language,
                speaker_wav=request.speaker_wav,
            )

            import scipy.io.wavfile as wavfile

            buffer = io.BytesIO()
            wavfile.write(buffer, rate=24000, data=wav)
            buffer.seek(0)

            yield buffer.read()
        except Exception as e:
            logger.error(f"Streaming TTS error: {e}")
            yield b""

    return StreamingResponse(
        generate(),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=tts.wav"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
