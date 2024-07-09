from fastapi import APIRouter, WebSocket, HTTPException, Request
import asyncio
import numpy as np
import base64
from queue import Queue
from processors import audio_processor
from chatbot import get_groq_response, transcribe_audio
from config import FPS, BUFFER_SIZE, SR

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
    buffer = []
    
    async def process_and_send_chunk(chunk):
        projected_vertices, audio_chunk = await asyncio.to_thread(audio_processor.process_chunk, chunk)
        audio = base64.b64encode(audio_chunk).decode('utf-8')
        
        # Calculate duration of audio chunk
        chunk_duration = len(audio_chunk) / SR
        
        return {
            "audio": audio,
            "frames": projected_vertices.tolist(),
            "duration": chunk_duration
        }

    while True:
        if not uploaded_audio_data_queue.empty():
            uploaded_audio_data = uploaded_audio_data_queue.get()
            
            transcription = await transcribe_audio(uploaded_audio_data)

            response_generator = get_groq_response(transcription)
            
            async for chatbot_response in response_generator:
                buffer.append(chatbot_response)
                
                if len(buffer) >= BUFFER_SIZE:
                    chunk = np.concatenate(buffer[:BUFFER_SIZE])
                    buffer = buffer[BUFFER_SIZE:]
                    
                    chunk_data = await process_and_send_chunk(chunk)
                    await websocket.send_json(chunk_data)
                
                # Precompute next chunk while current is playing
                if len(buffer) >= BUFFER_SIZE:
                    next_chunk = np.concatenate(buffer[:BUFFER_SIZE])
                    asyncio.create_task(process_and_send_chunk(next_chunk))
            
            # Process any remaining audio in the buffer
            if buffer:
                remaining_chunk = np.concatenate(buffer)
                chunk_data = await process_and_send_chunk(remaining_chunk)
                await websocket.send_json(chunk_data)
                buffer.clear()
        
        await asyncio.sleep(0.1)