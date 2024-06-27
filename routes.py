from fastapi import APIRouter, WebSocket, HTTPException, Request
import asyncio
import numpy as np
import base64
from queue import Queue
from processors import audio_processor
from chatbot import get_groq_response, transcribe_audio
from config import FPS

router = APIRouter()
uploaded_audio_data_queue = Queue()

@router.post("/upload/")
async def upload_audio(request: Request):
    try:
        audio_data = await request.body()
        audio_data = np.frombuffer(audio_data, dtype=np.float32)
        uploaded_audio_data_queue.put(audio_data)
        return {"status": "Audio uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {str(e)}")

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        if not uploaded_audio_data_queue.empty():
            uploaded_audio_data = uploaded_audio_data_queue.get()
            
            transcription = await transcribe_audio(uploaded_audio_data)
            chatbot_response = await get_groq_response(transcription)

            projected_vertices, audio_chunks = audio_processor.process(chatbot_response)
            
            frame_count = 0
            while frame_count < len(projected_vertices):
                points = projected_vertices[frame_count].tolist()
                await websocket.send_json({"type": "points", "points": points})
                if frame_count < len(audio_chunks):
                    audio_chunk = audio_chunks[frame_count]
                    await websocket.send_json({"type": "audio", "audio": base64.b64encode(audio_chunk).decode('utf-8')})
                frame_count += 1
                await asyncio.sleep(1/FPS)
        else:
            
            await asyncio.sleep(0.1)