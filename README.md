# SER

```
python3 stream_ser/export_model.py --model_name hubert_xlsr --output_dir /media/admin123/DataVoice/
```


```
python3 stream_ser/export_model.py --model_name wav2vec_xlsr --output_dir /media/admin123/DataVoice/
```


```
python3 test_models.py --model_name hubert_xlsr --checkpoint_dir /media/admin123/DataVoice/ckpt --test_data /media/admin123/DataVoice/valid.csv
```

```
python3 test_models.py --model_name wav2vec_xlsr --checkpoint_dir /media/admin123/DataVoice/ckpt --test_data /media/admin123/DataVoice/valid.csv
```