import torch
import torch.nn as nn
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
import numpy as np
import soundfile as sf
import librosa

from models.regression import RegressionHead


class SERModel(nn.Module):
    def __init__(self, wav2vec_model, model_type='large'):
        super().__init__()
        # pretrained_model_name = 'nguyenvulebinh/wav2vec2-base-vi'  # vnpt_w2v_16k  # facebook/wav2vec2-base  # nguyenvulebinh/wav2vec2-base-vi
        # print('Pretrained model:', pretrained_model_name)
        self.wav2vec2 = wav2vec_model
        self.wav2vec2.freeze_feature_encoder()

        if model_type == 'small':
            print('small')
            hidden_size = 768
        else:
            hidden_size = 1024  # large
            # print('large')
        self.classifier = RegressionHead(hidden_size=hidden_size)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return logits

if __name__ == "__main__":
    # Tạo model
    w2v_config = "/media/admin123/DataVoice/ckpt"
    checkpoint_path = "/media/admin123/DataVoice/ckpt/0-best.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    w2v_config = Wav2Vec2Config.from_pretrained(w2v_config)
    w2v_model = Wav2Vec2Model(w2v_config)
    model = SERModel(wav2vec_model=w2v_model)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # audio_path = "/home/admin123/Documents/vnpt/stream_ser/demo_audio.wav"
    # audio, audio_sr = sf.read(audio_path)
    # if audio_sr != 16000:
    #     audio = librosa.resample(y=audio, orig_sr=audio_sr, target_sr=16000)
    #     audio_sr = 16000
    # audio = np.expand_dims(audio, 0).astype(np.float32)
    # feature_extractor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    # audio = feature_extractor(audio,  sampling_rate=16000, return_tensors="pt").input_values
    # audio = audio.to(device)
    # with torch.no_grad():
    #     logits = model(audio)
    # print("Logits:", logits)

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

