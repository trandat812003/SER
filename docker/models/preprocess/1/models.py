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

        MODEL_PRETRAIN = "facebook/wav2vec2-base-960h"

        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_PRETRAIN)
        self.model = Wav2Vec2Model.from_pretrained(MODEL_PRETRAIN).to(self.device)

        model_config = json.loads(args['model_config'])
        output_config = pb_utils.get_output_config_by_name(model_config, "output")
        self.output_type = pb_utils.triton_string_to_numpy(output_config['data_type'])

    def execute(self, requests):
        responses = []
        for request in requests:
            # Nhận input là bytes (raw audio)
            audio_bytes = pb_utils.get_input_tensor_by_name(request, "AUDIO_RAW").as_numpy()[0]
            audio = np.frombuffer(audio_bytes, dtype=np.float32)
            # Nếu cần decode từ wav, dùng soundfile hoặc librosa
            # audio, sr = sf.read(io.BytesIO(audio_bytes))
            # Nếu cần resample:
            audio = librosa.resample(audio, orig_sr, self.sample_rate)
            # Chuẩn hóa shape (1, length)
            audio = np.expand_dims(audio, 0).astype(np.float32)
            out_tensor = pb_utils.Tensor("input_values", audio)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
