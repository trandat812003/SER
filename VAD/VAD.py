import numpy as np
import torch
import torchaudio
import soundfile as sf
import os

__all__ = ['VoiceActivityDetector']

class VoiceActivityDetector:
    def __init__(self, model_name="silero-vad", threshold=0.5, min_speech_duration=0.5, device="auto"):
        self.model_name = model_name
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        if model_name == "silero-vad":
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            (self.get_speech_timestamps, _, _, _, _) = utils
            self.model.to(self.device)
        else:
            raise ValueError(f"Không hỗ trợ model: {model_name}")

    def load_audio(self, wav_path, sr=16000):
        wav, orig_sr = torchaudio.load(wav_path)
        if orig_sr != sr:
            wav = torchaudio.functional.resample(wav, orig_sr, sr)
        wav = wav.mean(dim=0).numpy()  # mono
        return wav, sr

    def detect_voice_activity(self, audio, sr=16000):
        if self.model_name == "silero-vad":
            tensor = torch.from_numpy(audio).float().to(self.device)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            speech_timestamps = self.get_speech_timestamps(
                tensor, self.model,
                sampling_rate=sr,
                threshold=self.threshold
            )
            segments = [(seg['start'], seg['end']) for seg in speech_timestamps]
            return segments
        else:
            raise ValueError(f"Không hỗ trợ model: {self.model_name}")

    def process_audio(self, wav_path, output_dir):
        audio, sr = self.load_audio(wav_path)
        segments = self.detect_voice_activity(audio, sr)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        saved_files = []
        for idx, (start, end) in enumerate(segments):
            seg_audio = audio[start:end]
            out_path = os.path.join(output_dir, f"segment_{idx+1}.wav")
            sf.write(out_path, seg_audio, sr)
            saved_files.append(out_path)
        return saved_files 