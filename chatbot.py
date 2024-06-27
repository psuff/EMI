import json
import base64
import asyncio
import websockets
import aiohttp
from pydub import AudioSegment
import numpy as np
import io
import librosa
import soundfile as sf
from config import GROQ_API_KEY, ELEVENLABS_API_KEY, VOICE_ID, GROQ_API_URL, GROQ_WHISPER_URL, SR, ELEVENLABS_SAMPLE_RATE, BUFFER_SIZE, INITIAL_PROMPT

async def text_chunker(chunks):
    """Split text into chunks, ensuring to not break sentences."""
    splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
    buffer = ""
    async for text in chunks:
        if buffer.endswith(splitters):
            yield buffer + " "
            buffer = text
        elif text.startswith(splitters):
            yield buffer + text[0] + " "
            buffer = text[1:]
        else:
            buffer += text
    if buffer:
        yield buffer + " "

async def text_to_speech_input_streaming(voice_id, text_iterator):
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_turbo_v2"
    audio_queue = asyncio.Queue()
    
    async def listen(websocket):
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                if data.get("audio"):
                    audio_data = base64.b64decode(data["audio"])
                    audio_data = AudioSegment.from_mp3(io.BytesIO(audio_data))
                    audio_data = np.array(audio_data.get_array_of_samples()).astype(np.float32)
                    await audio_queue.put(audio_data)
                elif data.get('isFinal'):
                    break
            except websockets.exceptions.ConnectionClosed:
                break
        await audio_queue.put(None)  # Signal end of stream

    async def stream_audio():
        buffer = np.array([], dtype=np.float32)
        while True:
            if len(buffer) < BUFFER_SIZE:
                chunk = await audio_queue.get()
                if chunk is None:  # End of stream
                    if len(buffer) > 0:
                        yield buffer
                    break
                buffer = np.concatenate([buffer, chunk])
            output_chunk = buffer[:BUFFER_SIZE]
            buffer = buffer[BUFFER_SIZE:]
            yield output_chunk

    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
            "xi_api_key": ELEVENLABS_API_KEY,
        }))

        listen_task = asyncio.create_task(listen(websocket))

        async for text in text_chunker(text_iterator):
            await websocket.send(json.dumps({"text": text, "try_trigger_generation": True}))
        await websocket.send(json.dumps({"text": ""}))

        async for audio_chunk in stream_audio():
            if ELEVENLABS_SAMPLE_RATE != SR:
                audio_chunk = librosa.resample(audio_chunk, orig_sr=ELEVENLABS_SAMPLE_RATE, target_sr=SR)
            yield audio_chunk

        await listen_task

async def get_groq_response(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            GROQ_API_URL,
            headers=headers,
            json={
                "model": "llama3-70b-8192",
                "messages": [{"role": "system", "content": INITIAL_PROMPT}, {"role": "user", "content": prompt}],
                "stream": True
            }
        ) as response:
            async def text_iterator():
                async for line in response.content:
                    if line:
                        try:
                            decoded_line = line.decode('utf-8')
                            parts = decoded_line.split('data: ')
                            if len(parts) > 1:
                                data = json.loads(parts[1])
                                if 'choices' in data and len(data['choices']) > 0:
                                    content = data['choices'][0]['delta'].get('content', '')
                                    if content:
                                        yield content
                        except json.JSONDecodeError:
                            pass
                    elif response.content.at_eof():
                        break

            audio_chunks = []
            async for audio_chunk in text_to_speech_input_streaming(VOICE_ID, text_iterator()):
                audio_chunks.append(audio_chunk)
            
            return np.concatenate(audio_chunks)

async def transcribe_audio(audio_data, samplerate=16000):
    with io.BytesIO() as f:
        sf.write(f, audio_data, samplerate, format='wav')
        f.seek(0)
        
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('file', f, filename='audio.wav', content_type='audio/wav')
            data.add_field('model', 'whisper-large-v3')
            
            async with session.post(
                GROQ_WHISPER_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                data=data
            ) as response:
                result = await response.json()
                return result.get('text', '')
