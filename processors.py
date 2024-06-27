import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor
import librosa
import soundfile as sf
import io
from config import MODEL_PATH, PRETRAINED_A2M_CKPT, SR, FPS, WIDTH, HEIGHT, CHUNK_SIZE
from models import Audio2MeshModel
import math

class DataProcessor:
    def __init__(self, sampling_rate=16000):
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH, local_files_only=True)
        self._sampling_rate = sampling_rate

    def extract_feature(self, speech_array):
        input_value = np.squeeze(self._processor(speech_array, sampling_rate=self._sampling_rate).input_values)
        return input_value

class AudioProcessor:
    def __init__(self, model_path, a2m_ckpt_path, sr=16000, fps=30, width=512, height=512):
        self.sr = sr
        self.fps = fps
        self.width = width
        self.height = height

        self.model_path = model_path
        self.a2m_ckpt_path = a2m_ckpt_path
        self.model = self._load_model(model_path, a2m_ckpt_path)
        self.default_landmarks = np.load("default_landmarks.npy")

    def _load_model(self, model_path, a2m_ckpt_path):
        a2m_model = Audio2MeshModel(model_path)
        a2m_model.load_state_dict(torch.load(a2m_ckpt_path, map_location='cpu'), strict=False)
        a2m_model.eval()
        return a2m_model

    def _prepare_audio_feature(self, wav_file):
        data_preprocessor = DataProcessor(self.sr)
        input_value = data_preprocessor.extract_feature(wav_file)
        seq_len = math.ceil(len(input_value) / self.sr * self.fps)
        return {
            "audio_feature": input_value,
            "seq_len": seq_len
        }

    def _normalize_waveform(self, waveform):
        waveform = waveform / torch.max(torch.abs(waveform))
        return waveform

    def _project_3d_to_2d(self, pred):
        projected_vertices = pred[:, :, :2]
        projected_vertices[:, :, 1] = -projected_vertices[:, :, 1]
        
        # Normalize x axis
        min_x = np.min(projected_vertices[:, :, 0])
        max_x = np.max(projected_vertices[:, :, 0])
        projected_vertices[:, :, 0] = self.width * (projected_vertices[:, :, 0] - min_x) / (max_x - min_x)
        
        # Normalize y axis
        min_y = np.min(projected_vertices[:, :, 1])
        max_y = np.max(projected_vertices[:, :, 1])
        projected_vertices[:, :, 1] = self.width * (projected_vertices[:, :, 1] - min_y) / (max_y - min_y)
        
        return projected_vertices
    
    def _create_audio_chunk(self, data):
        buffer = io.BytesIO()
        sf.write(buffer, data.numpy(), SR, format='wav', subtype='PCM_16')
        buffer.seek(0)
        return buffer.read()

    def process(self, waveform):
        sample = self._prepare_audio_feature(waveform)
        sample['audio_feature'] = torch.from_numpy(sample['audio_feature']).float().unsqueeze(0)
            
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)
        waveform = self._normalize_waveform(waveform)

        with torch.no_grad():
            pred = self.model.infer(waveform, sample['seq_len'])
        pred = pred.squeeze().detach().cpu().numpy()
        pred = pred.reshape(pred.shape[0], -1, 3)
        pred = pred + self.default_landmarks

        projected_vertices = self._project_3d_to_2d(pred)

        audio_chunks = [self._create_audio_chunk(waveform[0][i:i+CHUNK_SIZE]) for i in range(0, len(waveform[0]), CHUNK_SIZE)]

        return projected_vertices, audio_chunks

audio_processor = AudioProcessor(model_path=MODEL_PATH,
                                 a2m_ckpt_path=PRETRAINED_A2M_CKPT,
                                 sr=SR,
                                 fps=FPS,
                                 width=WIDTH,
                                 height=HEIGHT)