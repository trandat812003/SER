import numpy as np
import soundfile as sf
import time
import os
import tritonclient.http as httpclient
from transformers import Wav2Vec2FeatureExtractor

# --- Cấu hình ---
TRITON_URL = "10008b0d9592.ngrok-free.app"
SAMPLE_RATE = 16000
VAD_DIR = "vad_output_silero"

processor = Wav2Vec2FeatureExtractor(
    sampling_rate=SAMPLE_RATE,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False
)

def infer_for_model(model_name, audio_files):
    client = httpclient.InferenceServerClient(url=TRITON_URL, ssl=True)
    outputs = []
    total_audio = 0.0
    start = time.time()

    for wav in audio_files:
        audio, sr = sf.read(os.path.join(VAD_DIR, wav), dtype="float32")
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"{wav}: sampling rate must be {SAMPLE_RATE}")
        audio = np.expand_dims(audio, 0).astype("float32")

        input_values = processor(audio[0], sampling_rate=sr, return_tensors="pt")["input_values"] \
            .numpy().astype("float32")
        total_audio += input_values.shape[1] / SAMPLE_RATE

        inp = httpclient.InferInput("input_values", input_values.shape, "FP32")
        inp.set_data_from_numpy(input_values)
        out = httpclient.InferRequestedOutput("output")

        result = client.infer(
            model_name=model_name,
            inputs=[inp],
            outputs=[out],
            model_version=""
        )
        outputs.append(result.as_numpy("output"))

    elapsed = time.time() - start
    return outputs, total_audio, elapsed

# --- Chuẩn bị ---
files = sorted([f for f in os.listdir(VAD_DIR) if f.endswith(".wav")])
hubert_res, total_audio, t0 = infer_for_model("hubert", files)
onnx_res, _, t1 = infer_for_model("hubert_onnx", files)

# --- In kết quả so sánh ---
print("Index |               Hubert output                |             ONNX output                |   L2 diff")
print("-" * 100)
for i, (a, b) in enumerate(zip(hubert_res, onnx_res)):
    l2 = np.linalg.norm(a - b)
    print(f"[{i}]   {a.flatten()}   {b.flatten()}   {l2:.6f}")

# --- Tính throughput ---
print("\nThroughput:")
print(f"  - hubert        : {total_audio/t0:.2f} audio‑s/giây")
print(f"  - hubert_onnx   : {total_audio/t1:.2f} audio‑s/giây")
