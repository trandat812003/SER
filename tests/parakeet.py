import os, time, asyncio, numpy as np, soundfile as sf
from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor
import torch
from tritonclient.http import InferInput, InferRequestedOutput

TRITON_URL = "4d78b93b9eae.ngrok-free.app"
COMBINED_MODEL = "combined"
VAD_DIR = "test_dataa"
SAMPLE_RATE = 16000
    

kwargs = {
    'sample_rate': 16000, 
    'normalize': 'per_feature', 
    'window_size': 0.025, 
    'window_stride': 0.01, 
    'window': 'hann', 
    'features': 128, 
    'n_fft': 512, 
    'log': True, 
    'frame_splicing': 1, 
    'dither': 1e-05, 
    'pad_to': 0, 
    'pad_value': 0.0
}
processor = AudioToMelSpectrogramPreprocessor(**kwargs)


async def infer_combined(client, audio_signal, length):
    inp_signal = InferInput("audio_signal", audio_signal.shape, "FP32")
    inp_length = InferInput("length",       length.shape,       "INT64")
    targets       = np.zeros((audio_signal.shape[0], 1), dtype=np.int32)
    target_length = np.ones((audio_signal.shape[0],),      dtype=np.int32)
    s1 = np.zeros((2, audio_signal.shape[0], 640), dtype=np.float32)
    s2 = np.zeros((2, 1, 640), dtype=np.float32)

    inp_targets = InferInput("targets",       targets.shape,       "INT32")
    inp_tlen    = InferInput("target_length", target_length.shape, "INT32")
    inp_s1      = InferInput("states.1",      s1.shape,            "FP32")
    inp_s2      = InferInput("onnx::Slice_3",  s2.shape,            "FP32")

    inp_signal.set_data_from_numpy(audio_signal)
    inp_length.set_data_from_numpy(length)
    inp_targets.set_data_from_numpy(targets)
    inp_tlen.set_data_from_numpy(target_length)
    inp_s1.set_data_from_numpy(s1)
    inp_s2.set_data_from_numpy(s2)

    return await client.infer(
        COMBINED_MODEL,
        inputs=[inp_signal, inp_length, inp_targets, inp_tlen, inp_s1, inp_s2],
        outputs=[InferRequestedOutput("final_outputs")]
    )

async def process_file(client, wav):
    audio, sr = sf.read(os.path.join(VAD_DIR, wav))
    assert sr == SAMPLE_RATE

    audio_signal = torch.tensor(audio).unsqueeze(0)
    length_tensor = torch.tensor([audio_signal.shape[1]], dtype=torch.long)

    audio_signal, length_tensor = processor(input_signal=audio_signal, length=length_tensor)

    audio_duration = audio.shape[0] / SAMPLE_RATE
    start_time = time.time()
    resp = await infer_combined(client, audio_signal.numpy(), length_tensor.numpy())
    end_time = time.time()
    infer_time = end_time - start_time

    rtf = audio_duration / infer_time if audio_duration > 0 else float('inf')
    out = resp.as_numpy("final_outputs")
    out = np.squeeze(out, axis=2) if out.ndim == 4 else out
    # decoded = decode_output(out, tokens)
    # print(decoded)
    print(f"{wav} → shape: {out.shape}, duration: {audio_duration:.2f}s, infer: {infer_time:.3f}s, RTFx: {rtf:.3f}")
    return rtf

async def main():
    from tritonclient.http.aio import InferenceServerClient
    
    audio_files = [f for f in os.listdir(VAD_DIR) if f.endswith('.wav')]
    total_rtf = 0

    client = InferenceServerClient(
        url=TRITON_URL,
        ssl=True,
        conn_limit=10,
        conn_timeout=1800
    )

    try:
        start = time.time()
        for wav in audio_files:
            rtf = await process_file(client, wav)
            total_rtf += rtf
        duration = time.time() - start
        avg_rtf = total_rtf / len(audio_files) if audio_files else 0
        print(f"Processed {len(audio_files)} files in {duration:.2f} seconds")
        print(f"RTFx: {avg_rtf:.3f}")
    finally:
        await client.close()
    # async with aio_http.InferenceServerClient(url=TRITON_URL, ssl=True, conn_limit=10, timeout=timeout) as client:
    #     start = time.time()
    #     for wav in audio_files:
    #         rtf = await process_file(client, wav, tokens)
    #         total_rtf += rtf
    #     duration = time.time() - start
    #     avg_rtf = total_rtf / len(audio_files) if audio_files else 0
    #     print(f"Processed {len(audio_files)} files in {duration:.2f} seconds")
    #     print(f"RTFx: {avg_rtf:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
