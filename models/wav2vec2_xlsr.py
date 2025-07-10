import torch
import torch.nn as nn
import soundfile as sf
import librosa
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

from models.regression import RegressionHead


class SERModelWav2Vec2(nn.Module):
    def __init__(self, pretrained_name="facebook/wav2vec2-large-xlsr-53", num_labels=2):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_name)
        self.wav2vec2.freeze_feature_encoder()
        self.classifier = RegressionHead(hidden_size=1024)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values=input_values)
        # Lấy hidden state cuối cùng (mean pooling)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        pooled = hidden_states.mean(dim=1)         # (batch, hidden)
        logits = self.classifier(pooled)
        return logits


if __name__ == "__main__":
    # Tạo model
    # w2v_config = "/media/admin123/DataVoice/ckpt"
    # checkpoint_path = "/media/admin123/DataVoice/ckpt/0-best.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # w2v_config = Wav2Vec2Config.from_pretrained(w2v_config)
    # w2v_model = Wav2Vec2Model(w2v_config)
    model = SERModelWav2Vec2()
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    audio_path = "/home/admin123/Documents/vnpt/stream_ser/demo_audio.wav"
    audio, audio_sr = sf.read(audio_path)
    if audio_sr != 16000:
        audio = librosa.resample(y=audio, orig_sr=audio_sr, target_sr=16000)
        audio_sr = 16000
    audio = np.expand_dims(audio, 0).astype(np.float32)
    
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )
    audio = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio = {key: value.to(device) for key, value in audio.items()}
    # breakpoint()
    with torch.no_grad():
        logits = model(**audio)
    print("Logits:", logits)