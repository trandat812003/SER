# SER

```
python3 stream_ser/export_model.py --model_name hubert_xlsr --output_dir /media/admin123/DataVoice/
```


```
python3 stream_ser/export_model.py --model_name wav2vec_xlsr --output_dir /media/admin123/DataVoice/
```


```
python3 test_models.py --model_name hubert_xlsr --checkpoint_dir /media/admin123/DataVoice/checkpoints/checkpoints --test_data /media/admin123/DataVoice/valid.csv
```

```
python3 test_models.py --model_name wav2vec_xlsr --checkpoint_dir /media/admin123/DataVoice/checkpoints/hubert --test_data /media/admin123/DataVoice/valid.csv
```

```
ngrok config add-authtoken 2sqQXDekU28hLOzNjBQTT991GcQ_2NGz72esojbHSUVwM3RpU

curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
  | tee /etc/apt/sources.list.d/ngrok.list \
  && apt update \
  && apt install ngrok
```

```
/usr/src/tensorrt/bin/trtexec

CUDA_VISIBLE_DEVICES=5 /opt/tritonserver/bin/tritonserver --model-repository=/home/jovyan/datnt/models/
```

```
/usr/src/tensorrt/bin/trtexec --onnx=/home/jovyan/datnt/models/hubert_onnx/1/model.onnx --minShapes=input_values:1x1000 --optShapes=input_values:4x16000 --maxShapes=input_values:4x1600000 --saveEngine=/home/jovyan/datnt/models/hubert/1/model.plan


/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/jovyan/datnt/models/hubert_onnx/1/model.onnx \
  --minShapes=input_values:1x16000 \
  --optShapes=input_values:4x16000 \
  --maxShapes=input_values:32x1600000 \
  --fp32 \
  --saveEngine=/home/jovyan/datnt/models/hubert/1/model.plan
```


./build/bin/sherpa-onnx-offline \
  --encoder=./parakeet/encoder.int8.onnx \
  --decoder=./parakeet/decoder.onnx \
  --joiner=./parakeet/joiner.onnx \
  --tokens=./parakeet/tokens.txt \
  --model-type=nemo_transducer \
  ./parakeet/test_wavs/demo_audio.wav