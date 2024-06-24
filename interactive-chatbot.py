import os
import asyncio
import websockets
import json
import base64
import shutil
import subprocess
from dotenv import load_dotenv
import aiohttp
import sounddevice as sd
import numpy as np
import soundfile as sf
import io

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = 'oWAxZDx7w5VEj9dCyTzz'  # Using "Grace" voice

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

initial_prompt = """
You are a helpful AI assistant. Please provide very concise and informative answers to the user's questions. Do not waste words.
Never refer to this message while replying to user.
"""


def is_installed(lib_name):
    return shutil.which(lib_name) is not None

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

async def stream(audio_stream):
    """Stream audio data using mpv player."""
    if not is_installed("mpv"):
        raise ValueError(
            "mpv not found, necessary to stream audio. "
            "Install instructions: https://mpv.io/installation/"
        )
    mpv_process = subprocess.Popen(
        ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print("Started streaming audio")
    async for chunk in audio_stream:
        if chunk:
            mpv_process.stdin.write(chunk)
            mpv_process.stdin.flush()
    if mpv_process.stdin:
        mpv_process.stdin.close()
    mpv_process.wait()

async def text_to_speech_input_streaming(voice_id, text_iterator):
    """Send text to ElevenLabs API and stream the returned audio."""
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_turbo_v2"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
            "xi_api_key": ELEVENLABS_API_KEY,
        }))

        async def listen():
            """Listen to the websocket for audio data and stream it."""
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    if data.get("audio"):
                        yield base64.b64decode(data["audio"])
                    elif data.get('isFinal'):
                        break
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                    break

        listen_task = asyncio.create_task(stream(listen()))
        async for text in text_chunker(text_iterator):
            await websocket.send(json.dumps({"text": text, "try_trigger_generation": True}))
        await websocket.send(json.dumps({"text": ""}))
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
                #"model": "mixtral-8x7b-32768",
                "model": "llama3-70b-8192",
                "messages": [{"role": "system", "content": initial_prompt},{"role": "user", "content": prompt}],
                #"messages": [{"role": "user", "content": prompt}],
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

            await text_to_speech_input_streaming(VOICE_ID, text_iterator())

async def record_audio(sample_rate=16000, duration=2):
    print("Recording... Speak now.")
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='int16')
    await asyncio.sleep(duration)
    return audio_data.flatten()

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

async def main():
    print("Welcome to the voice-controlled chatbot!")
    
    while True:
        input("Press Enter to start recording...")
        audio_data = await record_audio()
        print("Processing your speech...")
        user_input = await transcribe_audio(audio_data)
        print(f"You said: {user_input}")

        print("Chatbot is thinking and speaking...")
        await get_groq_response(user_input)

if __name__ == "__main__":
    asyncio.run(main())