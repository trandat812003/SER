import os, time, asyncio
import numpy as np
import soundfile as sf
import tritonclient.http.aio as aio_http
from tritonclient.http import InferInput, InferRequestedOutput

TRITON_URL = "4d78b93b9eae.ngrok-free.app" #https://4d78b93b9eae.ngrok-free.app/
ENC_MODEL = "encoder"
DEC_MODEL = "decoder"
JOIN_MODEL = "joiner"
VAD_DIR = "test_dataa"
SAMPLE_RATE = 16000

async def infer_encoder(client, audio_signal, length):
    inp0 = InferInput("audio_signal", audio_signal.shape, "FP32")
    inp1 = InferInput("length",        length.shape,       "INT64")
    inp0.set_data_from_numpy(audio_signal)
    inp1.set_data_from_numpy(length)
    resp = await client.infer(
        ENC_MODEL,
        inputs=[inp0, inp1],
        outputs=[InferRequestedOutput("outputs"), InferRequestedOutput("encoded_lengths")]
    )
    return resp.as_numpy("outputs")

async def infer_decoder(client, enc_out):
    targets = np.zeros((enc_out.shape[0], 1), dtype=np.int32)
    target_length = np.array([1] * enc_out.shape[0], dtype=np.int32)
    s1 = np.zeros((2, enc_out.shape[0], 640), dtype=np.float32)
    s2 = np.zeros((2, enc_out.shape[0], 640), dtype=np.float32)

    inputs = [
        InferInput("targets",          targets.shape, "INT32"),
        InferInput("target_length",    target_length.shape, "INT32"),
        InferInput("states.1",         s1.shape,       "FP32"),
        InferInput("onnx::Slice_3",    s2.shape,       "FP32"),
    ]
    for inp, arr in zip(inputs, [targets, target_length, s1, s2]):
        inp.set_data_from_numpy(arr)
    resp = await client.infer(
        DEC_MODEL,
        inputs=inputs,
        outputs=[InferRequestedOutput("outputs"), InferRequestedOutput("prednet_lengths"),
                 InferRequestedOutput("states"), InferRequestedOutput("162")]
    )
    return resp.as_numpy("outputs")

async def infer_joiner(client, enc_out, dec_out):
    inp_e = InferInput("encoder_outputs", enc_out.shape, "FP32")
    inp_d = InferInput("decoder_outputs", dec_out.shape, "FP32")
    inp_e.set_data_from_numpy(enc_out)
    inp_d.set_data_from_numpy(dec_out)
    resp = await client.infer(
        JOIN_MODEL,
        inputs=[inp_e, inp_d],
        outputs=[InferRequestedOutput("outputs")]
    )
    return resp.as_numpy("outputs")

async def process_file(client, wav):
    audio, sr = sf.read(os.path.join(VAD_DIR, wav), dtype='float32')
    assert sr == SAMPLE_RATE
    audio = np.expand_dims(audio, 0)
    length = np.array([audio.shape[1]], dtype=np.int64)
    # TODO: replace bằng feature extraction
    audio_signal = np.zeros((1, 128, audio.shape[1]), dtype=np.float32)
    audio_signal[0, :, :audio.shape[1]] = audio[0, :audio_signal.shape[2]]

    # breakpoint()

    enc_out = await infer_encoder(client, audio_signal, length)
    # breakpoint()
    dec_out = await infer_decoder(client, enc_out)
    join_out = await infer_joiner(client, enc_out, dec_out)
    print(f"{wav} → enc_out {enc_out.shape}, dec_out {dec_out.shape}, join_out {join_out.shape}")
    print(f"dec_out:{dec_out}, join_out: {join_out}")

async def main():
    files = [f for f in os.listdir(VAD_DIR) if f.endswith('.wav')]
    async with aio_http.InferenceServerClient(url=TRITON_URL, ssl=True, conn_limit=1) as client:
        start = time.time()
        for wav in files:
            await process_file(client, wav)
        print("Processed", len(files), "files in", time.time() - start, "seconds")

if __name__ == "__main__":
    asyncio.run(main())
