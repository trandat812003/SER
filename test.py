import os
import csv
import time
import numpy as np
import soundfile as sf
from sklearn.metrics import accuracy_score, f1_score
from transformers import Wav2Vec2FeatureExtractor
import tritonclient.http as httpclient

# ─── Cấu hình ─────────────────────────────────────────────────────
TRITON_URL = "10008b0d9592.ngrok-free.app"
MODEL = "hubert_onnx"
SAMPLE_RATE = 16000

processor = Wav2Vec2FeatureExtractor(
    sampling_rate=SAMPLE_RATE,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False
)

def mapping_pleasure_output(pleasure):
    if pleasure < -1.8:
        return 'extreme'
    elif pleasure < -0.5:
        return 'high'
    elif pleasure < -0.1:
        return 'medium'
    else:
        return 'low'

def mapping_arousal_output(arousal):
    if arousal > 1.5:
        return 'extreme'
    elif arousal > 0.5:
        return 'high'
    elif arousal > 0.01:
        return 'medium'
    else:
        return 'low'

def detect_unsatisfied(pleasure_output, arousal_output):
    if pleasure_output in ['extreme', 'high'] and arousal_output in ['extreme', 'high']:
        return 1
    elif pleasure_output in ['extreme', 'high'] and arousal_output in ['medium', 'low']:
        return 1
    else:
        return 0

# ─── Đọc CSV và inference ────────────────────────────────────────
csv_path = "/media/admin123/DataVoice/valid.csv"  # Đường dẫn tới file CSV (tab-separated)

y_true = []
y_pred = []

client = httpclient.InferenceServerClient(url=TRITON_URL, ssl=True)

with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        wav = row['audio']
        pleasure_gt = float(row['pleasure'])
        arousal_gt = float(row['arousal'])

        filepath = os.path.join("/media/admin123/DataVoice/SER/audio/audio_2500", wav)
        audio, sr = sf.read(filepath, dtype='float32')
        assert sr == SAMPLE_RATE, "Sai sample rate!"
        audio = np.expand_dims(audio, 0).astype('float32')

        input_values = processor(audio[0], sampling_rate=sr, return_tensors="pt")["input_values"].numpy().astype("float32")

        inp = httpclient.InferInput("input_values", input_values.shape, "FP32")
        inp.set_data_from_numpy(input_values)
        out_req = httpclient.InferRequestedOutput("output")

        res = client.infer(
            model_name=MODEL,
            inputs=[inp],
            outputs=[out_req],
            model_version=""
        )
        output = res.as_numpy("output").flatten()
        p_pred_raw, a_pred_raw = float(output[0]), float(output[1])

        p_pred = mapping_pleasure_output(p_pred_raw)
        a_pred = mapping_arousal_output(a_pred_raw)
        unsat_pred = detect_unsatisfied(p_pred, a_pred)

        p_gt = mapping_pleasure_output(pleasure_gt)
        a_gt = mapping_arousal_output(arousal_gt)
        unsat_gt = detect_unsatisfied(p_gt, a_gt)

        y_true.append(unsat_gt)
        y_pred.append(unsat_pred)
acc = accuracy_score(y_true, y_pred)

print("\n===== KẾT QUẢ =====")
print(f"Accuracy: {acc*100:.2f}%")
