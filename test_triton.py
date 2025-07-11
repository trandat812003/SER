import numpy as np
import soundfile as sf
import tritonclient.http as httpclient

audio, sr = sf.read("demo_audio.wav", dtype='float32')
audio = np.expand_dims(audio, 0).astype(np.float32)

client = httpclient.InferenceServerClient(url="localhost:8000")
inputs = [httpclient.InferInput("AUDIO_RAW", audio.shape, "FP32")]
inputs[0].set_data_from_numpy(audio)
outputs = [httpclient.InferRequestedOutput("output")]
results = client.infer("ensemble_ser", inputs=inputs, outputs=outputs)
print("Kết quả:", results.as_numpy("output"))
