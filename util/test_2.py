# from tqdm import tqdm
# from torch.utils.data import DataLoader
from nemo.collections.common import tokenizers

# # from nemo.core.classes.common import Serialization
# from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
# from nemo.collections.asr.data.audio_to_text_dataset import get_bpe_dataset
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding
from omegaconf import DictConfig, OmegaConf, open_dict, ListConfig
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.modules import RNNTDecoder, RNNTJoint
import torch

tokenizer = tokenizers.SentencePieceTokenizer(
    model_path="/home/admin123/Downloads/parakeet-tdt-0.6b-v2/705f11d22dc04b169effc35ce5cd1361_tokenizer.model"
)

vocabulary = {}
for i in range(tokenizer.vocab_size):
    piece = tokenizer.ids_to_tokens([i])
    piece = piece[0]
    vocabulary[piece] = i + 1


def get_vocab():
    return vocabulary


tokenizer.tokenizer.vocab_size = len(vocabulary)
tokenizer.tokenizer.get_vocab = vocabulary
tokenizer.tokenizer.all_special_tokens = tokenizer.special_token_to_id

decoder_cfg = torch.load(f"decoder_config.pt")
joint_cfg = torch.load(f"joint_config.pt")

decoder = RNNTDecoder(**decoder_cfg)
joint = RNNTJoint(**joint_cfg)

decoder.load_state_dict(torch.load(f"decoder.ckpt"))
joint.load_state_dict(torch.load(f"joint.ckpt"))

cfg_decoding = {
    "model_type": "tdt",
    "strategy": "greedy_batch",
    "compute_hypothesis_token_set": False,
    "preserve_alignments": True,
    "tdt_include_token_duration": None,
    "confidence_cfg": {
        "preserve_frame_confidence": False,
        "preserve_token_confidence": False,
        "preserve_word_confidence": False,
        "exclude_blank": True,
        "aggregation": "min",
        "tdt_include_duration": False,
        "method_cfg": {
            "name": "entropy",
            "entropy_type": "tsallis",
            "alpha": 0.33,
            "entropy_norm": "exp",
            "temperature": "DEPRECATED",
        },
    },
    "fused_batch_size": None,
    "compute_timestamps": True,
    "compute_langs": False,
    "word_seperator": " ",
    "segment_seperators": [".", "!", "?"],
    "segment_gap_threshold": None,
    "rnnt_timestamp_type": "all",
    "greedy": {
        "max_symbols_per_step": 10,
        "preserve_alignments": False,
        "preserve_frame_confidence": False,
        "tdt_include_token_duration": False,
        "tdt_include_duration_confidence": False,
        "confidence_method_cfg": {
            "name": "entropy",
            "entropy_type": "tsallis",
            "alpha": 0.33,
            "entropy_norm": "exp",
            "temperature": "DEPRECATED",
        },
        "loop_labels": True,
        "use_cuda_graph_decoder": True,
        "ngram_lm_model": None,
        "ngram_lm_alpha": 0.0,
        "max_symbols": 10,
    },
    "beam": {
        "beam_size": 2,
        "search_type": "default",
        "score_norm": True,
        "return_best_hypothesis": False,
        "tsd_max_sym_exp_per_step": 50,
        "alsd_max_target_len": 2.0,
        "nsc_max_timesteps_expansion": 1,
        "nsc_prefix_alpha": 1,
        "maes_num_steps": 2,
        "maes_prefix_alpha": 1,
        "maes_expansion_gamma": 2.3,
        "maes_expansion_beta": 2,
        "language_model": None,
        "softmax_temperature": 1.0,
        "preserve_alignments": False,
        "ngram_lm_model": None,
        "ngram_lm_alpha": 0.0,
        "hat_subtract_ilm": False,
        "hat_ilm_weight": 0.0,
        "max_symbols_per_step": 10,
        "allow_cuda_graphs": True,
        "tsd_max_sym_exp": 50,
    },
    "temperature": 1.0,
    "durations": [0, 1, 2, 3, 4],
    "big_blank_durations": [],
}
cfg_decoding = OmegaConf.create(cfg_decoding)
decoding = RNNTBPEDecoding(
    decoding_cfg=cfg_decoding,
    decoder=decoder,
    joint=joint,
    tokenizer=tokenizer,
)

from nemo.collections.asr.parts.utils.timestamp_utils import process_timestamp_outputs


def transcribe_output_processing(encoded, encoded_len):
    hyp = decoding.rnnt_decoder_predictions_tensor(
        encoded,
        encoded_len,
        return_hypotheses=True,
    )
    # cleanup memory
    del encoded, encoded_len

    hyp = process_timestamp_outputs(hyp, 8, 0.01)

    return hyp


import numpy as np

outputs_triton = np.load("outputs_triton.npy")
outputs_triton = torch.from_numpy(outputs_triton)

hyp = transcribe_output_processing(
    outputs_triton, torch.tensor([751], dtype=torch.long)
)

segment_timestamps = hyp[0].timestamp.get("segment", [])
word_timestamps = hyp[0].timestamp.get("word", [])
char_timestamps = hyp[0].timestamp.get("char", [])

# Print segment-level timestamps
print("\n[Segment-level timestamps]")
for stamp in segment_timestamps:
    print(f"{stamp['start']}s - {stamp['end']}s : {stamp['segment']}")
