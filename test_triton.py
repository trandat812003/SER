import numpy as np
import soundfile as sf
import tritonclient.http as httpclient

# Đọc file audio và chuyển thành bytes
audio, sr = sf.read("demo_audio.wav")
audio = audio.astype(np.float32)
audio_bytes = audio.tobytes()
audio_np = np.array([np.frombuffer(audio_bytes, dtype=np.uint8)])

client = httpclient.InferenceServerClient(url="localhost:8000")
inputs = [httpclient.InferInput("AUDIO_RAW", audio_np.shape, "UINT8")]
inputs[0].set_data_from_numpy(audio_np)
outputs = [httpclient.InferRequestedOutput("output")]
results = client.infer("ensemble_ser", inputs, outputs=outputs)
print("Kết quả:", results.as_numpy("output"))