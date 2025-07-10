import io
import librosa
import numpy as np
import soundfile as sf
import wave
import time
import torch
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor

from VAD.VAD import VoiceActivityDetector
from models.wav2vec2_xlsr import SERModelWav2Vec2
from models.hubert_xlsr import SERModelHuBERT
from models.wav2vec2 import SERModel

class VNPTStreamEmotion:
    def __init__(self, model_name="wav2vec2_xlsr", model_ckpt=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == "wav2vec2_xlsr":
            self.model = SERModelWav2Vec2()
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        elif model_name == "hubert_xlsr":
            self.model = SERModelHuBERT()
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
        elif model_name == "wav2vec2":
            w2v_config = "/media/admin123/DataVoice/ckpt"
            checkpoint_path = "/media/admin123/DataVoice/ckpt/0-best.pth"
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            w2v_config = Wav2Vec2Config.from_pretrained(w2v_config)
            w2v_model = Wav2Vec2Model(w2v_config)
            self.model = SERModel(wav2vec_model=w2v_model)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint)
            self.model.to(device)
            self.model.eval()
            self.feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0.0,
                do_normalize=True,
                return_attention_mask=False,
            )
        else:
            raise ValueError("Không hỗ trợ model: " + model_name)
        if model_ckpt:
            self.model.load_state_dict(torch.load(model_ckpt, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.vad = VoiceActivityDetector()

    def inference_file(self, audio_path, id=None):
        # 1. Tách voice bằng VAD
        segments = self.vad.process_audio(audio_path, output_dir="vad_tmp")
        results = []
        for idx, seg_path in enumerate(segments):
            # 2. Load audio segment
            audio, sr = sf.read(seg_path)
            if sr != 16000:
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)
            audio = np.expand_dims(audio, 0).astype(np.float32)
            # 3. Extract feature
            inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            input_values = inputs["input_values"].to(self.device)
            # 4. Dự đoán cảm xúc
            with torch.no_grad():
                logits = self.model(input_values)
                pred = torch.argmax(logits, dim=-1).item()
            results.append({
                "id": id,
                "segment": idx+1,
                "emotion": pred,
                "segment_path": seg_path
            })
        return results

if __name__ == "__main__":
    # Ví dụ: chọn model wav2vec2_xlsr, có thể đổi thành "hubert_xlsr" hoặc "wav2vec2"
    sermodel = VNPTStreamEmotion(model_name="wav2vec2")
    for result in sermodel.inference_file("/home/admin123/Documents/vnpt/stream_ser/demo_audio.wav", id="1"):
        print(result)
