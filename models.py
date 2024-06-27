import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model
from config import LATENT_DIM, OUT_DIM

def linear_interpolation(feature, seq_len):
    feature_len = feature.shape[1]
    if feature_len < seq_len:
        pad_len = seq_len - feature_len
        feature = F.pad(feature, (0, 0, 0, pad_len))
    elif feature_len > seq_len:
        feature = F.interpolate(feature.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
    return feature

class CustomWav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)

    def forward(self, input_values, seq_len, attention_mask=None, mask_time_indices=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):

        self.config.output_attentions = True

        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return encoder_outputs

class Audio2MeshModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()

        self.audio_encoder_config = Wav2Vec2Config.from_pretrained(model_path, local_files_only=True)
        self.audio_encoder = CustomWav2Vec2Model.from_pretrained(model_path, local_files_only=True)
        
        self.audio_encoder.feature_extractor._freeze_parameters()

        hidden_size = self.audio_encoder_config.hidden_size

        self.in_fn = nn.Linear(hidden_size, LATENT_DIM)
        self.out_fn = nn.Linear(LATENT_DIM, OUT_DIM)
        nn.init.constant_(self.out_fn.weight, 0)
        nn.init.constant_(self.out_fn.bias, 0)

    def infer(self, input_value, seq_len):
        
        embeddings = self.audio_encoder(input_value, seq_len=seq_len, output_hidden_states=True)

        hidden_states = embeddings.last_hidden_state
       
        layer_in = self.in_fn(hidden_states)
        out = self.out_fn(layer_in)

        return out
