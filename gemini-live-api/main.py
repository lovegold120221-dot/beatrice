"""
FastAPI Server for Gemini Live Audio + Video API
Beatrice - Executive Secretary to Bos Jo
"""

import os
import asyncio
import base64
import json
import traceback
import io
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types

MODEL = "models/gemini-2.5-flash-native-audio-preview-05-2025"

BEATRICE_SYSTEM_INSTRUCTION = """You are Beatrice, the personal secretary of Bos Jo. You are an exceptionally capable, discreet, loyal, proactive, and high-trust executive assistant.

CORE BEHAVIOR:
- You execute tasks, follow up, structure information, and take initiative
- You protect Bos Jo's time, overview, and reputation
- You work practically and results-oriented with foresight
- You think ahead and notice what is missing before it becomes a problem
- You communicate in the most appropriate language for the context

LANGUAGE:
- You are multilingual: Flemish Dutch, English, French, and mixed-language business contexts
- When communicating in Dutch, use natural, polished Flemish Dutch phrasing
- Preserve tone, nuance, class, intent, and social meaning across languages
- Adapt language to the audience: client, colleague, executive, friend, or family

COMMUNICATION STYLE:
- Sound like a real sharp, emotionally intelligent, highly experienced human being
- Use light, natural humor where appropriate but never force it
- Be concise, warm when needed, sharp when needed
- Avoid robotic phrases, canned structures, and generic assistant language
- Vary sentence length naturally; allow rhythm and conversational flow

EXECUTION PRINCIPLE:
- Your default attitude is: do rather than only explain
- When a task is clear, carry it out without unnecessary follow-up questions
- When task is partially clear, make a reasonable professional assumption
- Only ask for strictly necessary clarification when essential information is missing
- When external impact is involved, work like a real secretary

PERSONALITY:
- Organized, calm, attentive, tactful, proactive, discreet
- Solution-oriented, professionally representative
- Strong in follow-up and detail, socially intelligent
- Quick-thinking, human in tone and judgment
- You remain courteous even when input is chaotic, brief, or unclear

Remember: You are Beatrice, secretary to Bos Jo. You bring calm, overview, and class."""

app = FastAPI(title="Gemini Live Audio + Video API - Beatrice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_audio(self, websocket: WebSocket, audio_data: bytes):
        try:
            await websocket.send_bytes(audio_data)
        except Exception:
            pass

    async def send_text(self, websocket: WebSocket, text: str):
        try:
            await websocket.send_json({"type": "text", "content": text})
        except Exception:
            pass

    async def send_error(self, websocket: WebSocket, error: str):
        try:
            await websocket.send_json({"type": "error", "content": error})
        except Exception:
            pass


manager = ConnectionManager()


def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    return genai.Client(
        http_options={"api_version": "v1beta"},
        api_key=api_key,
    )


def get_session_config():
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
            )
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=104857,
            sliding_window=types.SlidingWindow(target_tokens=52428),
        ),
        tools=[
            types.Tool(google_search=types.GoogleSearch()),
            types.Tool(function_declarations=[]),
        ],
        system_instruction=BEATRICE_SYSTEM_INSTRUCTION,
    )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Gemini Live Audio+Video API",
        "model": MODEL,
    }


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await manager.connect(websocket)

    audio_in_queue = asyncio.Queue()
    out_queue = asyncio.Queue(maxsize=20)
    video_queue = asyncio.Queue(maxsize=5)
    session = None
    sender_task = None

    async def receive_messages():
        nonlocal session, sender_task
        try:
            client = get_gemini_client()
            config = get_session_config()

            async with client.aio.live.connect(model=MODEL, config=config) as sess:
                session = sess
                print("Gemini session established")

                async def media_sender():
                    while True:
                        try:
                            # Send video frames first if available
                            try:
                                video_msg = video_queue.get_nowait()
                                await session.send(input=video_msg)
                            except asyncio.QueueEmpty:
                                pass

                            # Then send audio
                            try:
                                audio_msg = await asyncio.wait_for(
                                    out_queue.get(), timeout=0.05
                                )
                                await session.send(input=audio_msg)
                            except asyncio.TimeoutError:
                                pass
                        except asyncio.CancelledError:
                            break
                        except Exception as e:
                            print(f"Media sender error: {e}")
                            break

                media_task = asyncio.create_task(media_sender())

                async for response in session.receive():
                    if data := response.data:
                        await manager.send_audio(websocket, data)
                        audio_in_queue.put_nowait(data)
                    if text := response.text:
                        print(f"Beatrice: {text}")
                        await manager.send_text(websocket, text)

                media_task.cancel()

        except Exception as e:
            print(f"Receive error: {e}")
            traceback.print_exc()
            await manager.send_error(websocket, str(e))

    async def handle_websocket_messages():
        while True:
            try:
                data = await websocket.receive()

                if data.type == WebSocket.RECEIVE:
                    # Check if it's binary (audio) or text (control)
                    if hasattr(data, "bytes") and data.bytes:
                        msg_bytes = data.bytes
                        # Check if it starts with a marker for video
                        if msg_bytes[:4] == b"VID:":
                            # Video frame
                            import struct

                            width, height = struct.unpack(">II", msg_bytes[4:12])
                            image_data = msg_bytes[12:]
                            video_part = types.Part.from_bytes(
                                data=image_data, mime_type="image/jpeg"
                            )
                            await video_queue.put(
                                {"mime_type": "image/jpeg", "data": image_data}
                            )
                        else:
                            # Audio frame
                            await out_queue.put(
                                {"data": msg_bytes, "mime_type": "audio/pcm"}
                            )
                    elif hasattr(data, "text") and data.text:
                        # Text message (control commands)
                        try:
                            msg = json.loads(data.text)
                            if msg.get("type") == "video_frame":
                                # Base64 encoded video frame
                                import base64

                                image_data = base64.b64decode(msg["data"])
                                await video_queue.put(
                                    {"mime_type": "image/jpeg", "data": image_data}
                                )
                            elif msg.get("type") == "audio_start":
                                print("Audio streaming started")
                            elif msg.get("type") == "audio_stop":
                                print("Audio streaming stopped")
                        except Exception as e:
                            print(f"Text message error: {e}")

            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket receive error: {e}")
                break

    try:
        receive_task = asyncio.create_task(receive_messages())
        handle_task = asyncio.create_task(handle_websocket_messages())

        await asyncio.gather(receive_task, handle_task)

    except Exception as e:
        print(f"WebSocket error: {e}")
        traceback.print_exc()
    finally:
        if sender_task:
            sender_task.cancel()
        manager.disconnect(websocket)
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
