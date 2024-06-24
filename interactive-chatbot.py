import os
from dotenv import load_dotenv
import requests
import sounddevice as sd
import numpy as np
import soundfile as sf
import io

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech/oWAxZDx7w5VEj9dCyTzz/stream"  # Using "Grace" voice

headers_groq = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

headers_elevenlabs = {
    "xi-api-key": ELEVENLABS_API_KEY,
    "Content-Type": "application/json"
}

def get_groq_response(prompt):
    response = requests.post(
        GROQ_API_URL,
        headers=headers_groq,
        json={
            "model": "mixtral-8x7b-32768",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 1000
        }
    )
    response_data = response.json()
    return response_data['choices'][0]['message']['content']

def get_elevenlabs_audio(text):
    response = requests.post(
        ELEVENLABS_API_URL,
        headers=headers_elevenlabs,
        json={"text": text, "model_id": "eleven_multilingual_v1"}
    )

    if response.status_code == 200:
        audio_buffer = io.BytesIO(response.content)
        audio_data, samplerate = sf.read(audio_buffer, dtype='int16')
        return audio_data.astype(np.int16)
    else:
        print("Error:", response.status_code, response.text)
        return None

def play_audio(audio_data):
    with sd.OutputStream(samplerate=44100, channels=1, dtype='int16') as stream:
        stream.write(audio_data)

def record_audio(sample_rate=16000, duration=5):
    print("Recording... Speak now.")
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    return audio_data.flatten()

def transcribe_audio(audio_data, samplerate=16000):
    sf.write("tmp.wav", audio_data, samplerate)
    
    with open("tmp.wav", 'rb') as f:
        response = requests.post(
            GROQ_WHISPER_URL,
            headers={"Authorization": f"bearer {GROQ_API_KEY}"},
            files={'file': f}, 
            data={'model': 'whisper-large-v3'}
        )
    
    os.remove("tmp.wav")
    result = response.json()
    return result.get('text', '')

def main():
    print("Welcome to the voice-controlled chatbot!")
    while True:
        input("Press Space to start recording...")
        audio_data = record_audio()
        print("Processing your speech...")
        user_input = transcribe_audio(audio_data)
        print(f"You said: {user_input}")

        print("Chatbot is thinking...")
        response = get_groq_response(user_input)
        print(f"Chatbot: {response}")

        print("Chatbot is speaking...")
        audio_data = get_elevenlabs_audio(response)
        if audio_data is not None:
            play_audio(audio_data)

if __name__ == "__main__":
    main()