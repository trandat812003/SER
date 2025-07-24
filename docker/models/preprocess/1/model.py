import torch
import json
import numpy as np
from transformers import Wav2Vec2FeatureExtractor

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device
            ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.sample_rate = 16000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.sample_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
        )
        model_config = json.loads(args['model_config'])
        output_config = pb_utils.get_output_config_by_name(model_config, "input_values")
        self.output_type = pb_utils.triton_string_to_numpy(output_config['data_type'])

    def execute(self, requests):
        responses = []
        print(f"Số lượng request trong batch: {len(requests)}")
        # Nếu dynamic batching, mỗi request là 1 sample trong batch
        # Ghép tất cả input thành 1 mảng batch
        batch_audio = []
        for request in requests:
            audio_bytes = pb_utils.get_input_tensor_by_name(request, "AUDIO_RAW").as_numpy()
            batch_audio.append(audio_bytes)
        batch_audio = np.concatenate(batch_audio, axis=0)  # shape: (batch, length)
        # Xử lý batch với processor
        audio = self.processor(batch_audio, sampling_rate=16000, return_tensors="pt")['input_values']
        audio = audio.cpu().detach().numpy()  # shape: (batch, length)
        # Trả về batch output cho từng request
        for i in range(len(requests)):
            out_tensor = pb_utils.Tensor("input_values", audio[i:i+1])  # giữ shape (1, length)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
