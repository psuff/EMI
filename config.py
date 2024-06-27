import os
    
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = 'oWAxZDx7w5VEj9dCyTzz'  # Using "Grace" voice

# Global parameters
WIDTH = 512
HEIGHT = 512
FPS = 30
SR = 16000
CHUNK_SIZE = int(SR / FPS)

# Model parameters
OUT_DIM = 1404
LATENT_DIM = 512
MODEL_PATH = './pretrained_model/wav2vec2-base-960h'

# File paths
PRETRAINED_A2M_CKPT = './pretrained_model/audio2mesh.pt'

ELEVENLABS_SAMPLE_RATE = 44100  # ElevenLabs typically uses 44.1kHz
BUFFER_SIZE = 2048

INITIAL_PROMPT = "Prentend to be a real human, with emotions and feelings. \
Your goal is to keep the conversation going, so you always have to make sure that the user is engaged and wants to keep the conversation going. \
Please provide very concise and informative answers to the user's questions. Do not waste words. \
Never refer to the fact that you are an AI, LLM, etc. Always pretend to be a friendly empathetic human."