# from tqdm import tqdm
# from torch.utils.data import DataLoader
from nemo.collections.common import tokenizers
# # from nemo.core.classes.common import Serialization
# from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
# from nemo.collections.asr.data.audio_to_text_dataset import get_bpe_dataset
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding
from omegaconf import DictConfig, OmegaConf, open_dict,ListConfig
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.losses.rnnt import RNNTLoss, resolve_rnnt_default_loss_name
import torch

tokenizer = tokenizers.SentencePieceTokenizer(model_path="/home/admin123/Downloads/parakeet-tdt-0.6b-v2/705f11d22dc04b169effc35ce5cd1361_tokenizer.model")

cfg = OmegaConf.load("config.yaml")

vocabulary = {}
for i in range(tokenizer.vocab_size):
    piece = tokenizer.ids_to_tokens([i])
    piece = piece[0]
    vocabulary[piece] = i + 1

# wrapper method to get vocabulary conveniently
def get_vocab():
    return vocabulary

# attach utility values to the tokenizer wrapper
tokenizer.tokenizer.vocab_size = len(vocabulary)
tokenizer.tokenizer.get_vocab = get_vocab
tokenizer.tokenizer.all_special_tokens = tokenizer.special_token_to_id

vocabulary = tokenizer.tokenizer.get_vocab()

# Set the new vocabulary
with open_dict(cfg):
    cfg.labels = ListConfig(list(vocabulary))

with open_dict(cfg.decoder):
    cfg.decoder.vocab_size = len(cfg.labels)

with open_dict(cfg.joint):
    cfg.joint.num_classes = len(cfg.labels)
    cfg.joint.vocabulary = cfg.labels
    cfg.joint.jointnet.encoder_hidden = cfg.model_defaults.enc_hidden
    cfg.joint.jointnet.pred_hidden = cfg.model_defaults.pred_hidden

vocabulary = {}
for i in range(tokenizer.vocab_size):
    piece = tokenizer.ids_to_tokens([i])
    piece = piece[0]
    vocabulary[piece] = i + 1

tokenizer.tokenizer.vocab_size = len(vocabulary)
tokenizer.tokenizer.get_vocab = get_vocab
tokenizer.tokenizer.all_special_tokens = tokenizer.special_token_to_id

# config = {'manifest_filepath': '/home/admin123/Downloads/manifest.json', 'sample_rate': 16000, 'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'pin_memory': True, 'channel_selector': None, 'use_start_end_token': False}
# dataset = get_bpe_dataset(config=config, tokenizer=tokenizer, augmentor=None)

# # config_preprocess = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'sample_rate': 16000, 'normalize': 'per_feature', 'window_size': 0.025, 'window_stride': 0.01, 'window': 'hann', 'features': 128, 'n_fft': 512, 'log': True, 'frame_splicing': 1, 'dither': 1e-05, 'pad_to': 0, 'pad_value': 0.0}
# # preprocessor = Serialization.from_config_dict(config_preprocess)

# SAMPLE_RATE = 16000

# preprocessor = AudioToMelSpectrogramPreprocessor(
#     sample_rate=16000,
#     normalize="per_feature",
#     window_size=0.025,
#     window_stride=0.01,
#     window="hann",
#     features=128,
#     n_fft=512,
#     log=True,
#     frame_splicing=1,
#     dither=1e-05,
#     pad_to=0,
#     pad_value=0.0,
# )

# dataloader = DataLoader(
#     dataset=dataset,
#     batch_size=1,
#     collate_fn=dataset.collate_fn,
# )

# for test_batch in tqdm(dataloader, desc="Transcribing"):
#     test_batch = preprocessor(
#         input_signal=test_batch[0],
#         length=test_batch[1]
#     )
#     breakpoint()
#     print(test_batch)

decoder = EncDecRNNTModel.from_config_dict(cfg.decoder)
joint = EncDecRNNTModel.from_config_dict(cfg.joint)

decoder.load_state_dict(torch.load("decoder.ckpt"))
joint.load_state_dict(torch.load("joint.ckpt"))

cfg_decoding = {'strategy': 'greedy_batch', 'model_type': 'tdt', 'durations': [0, 1, 2, 3, 4], 'greedy': {'max_symbols': 10}, 'beam': {'beam_size': 2, 'return_best_hypothesis': False, 'score_norm': True, 'tsd_max_sym_exp': 50, 'alsd_max_target_len': 2.0}}
cfg_decoding = OmegaConf.create(cfg_decoding)
decoding = RNNTBPEDecoding(
    decoding_cfg=cfg_decoding,
    decoder=decoder,
    joint=joint,
    tokenizer=tokenizer,
)

from typing import Any, Dict, List, Optional, Union
from nemo.collections.asr.parts.utils.timestamp_utils import process_timestamp_outputs

def transcribe_output_processing(encoded, encoded_len):
    hyp = decoding.rnnt_decoder_predictions_tensor(
        encoded,
        encoded_len,
    )
    # cleanup memory
    del encoded, encoded_len

    breakpoint()

    hyp = process_timestamp_outputs(
        hyp, 8, cfg['preprocessor']['window_stride']
    )

    return hyp


import numpy as np
outputs_triton = np.load("outputs_triton.npy")
outputs_triton = torch.from_numpy(outputs_triton)
breakpoint()

hyp = transcribe_output_processing(outputs_triton, torch.tensor([751], dtype=torch.long))


