import os, time, asyncio, numpy as np
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor
from tritonclient.http import InferInput, InferRequestedOutput
import tritonclient.http.aio as aioclient

TRITON_URL = "4d78b93b9eae.ngrok-free.app"
MODEL = "hubert"
SAMPLE_RATE = 16000
VAD_DIR = "vad_output_silero"

processor = Wav2Vec2FeatureExtractor(
    sampling_rate=SAMPLE_RATE,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False
)

async def infer_one(client, np_audio, file_name):
    infer_input = InferInput("input_values", np_audio.shape, "FP32")
    infer_input.set_data_from_numpy(np_audio)

    infer_output = InferRequestedOutput("output")

    try:
        result = await client.infer(
            model_name=MODEL,
            inputs=[infer_input],
            model_version="",
            outputs=[infer_output]
        )
        output_array = result.as_numpy("output")
        print(f"{file_name}: {output_array}")
        return result
    except Exception as e:
        print(f"❌ Lỗi file {file_name}: {e}")
        return None

async def main():
    audio_files = [f for f in os.listdir(VAD_DIR) if f.endswith('.wav')]
    total_audio_sec = 0.0

    async with aioclient.InferenceServerClient(url=TRITON_URL, ssl=True) as client:
        tasks = []
        for wav in audio_files:
            audio, sr = sf.read(os.path.join(VAD_DIR, wav), dtype='float32')
            assert sr == SAMPLE_RATE
            audio = np.expand_dims(audio, 0).astype('float32')
            inputs = processor(audio[0], sampling_rate=sr, return_tensors="pt")
            input_values = inputs["input_values"].numpy().astype('float32')
            total_audio_sec += input_values.shape[1] / SAMPLE_RATE
            tasks.append(infer_one(client, input_values, wav))

        start = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start

        print(f"Tổng audio: {total_audio_sec:.2f}s")
        print(f"Thời gian inference: {elapsed:.2f}s")
        print(f"Throughput: {total_audio_sec/elapsed:.2f} audio‑s/giây")

if __name__ == "__main__":
    asyncio.run(main())
