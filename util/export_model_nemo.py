import os
import nemo.collections.asr as nemo_asr

# 1. Load model
model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
model.eval()
model.to('cpu')

# 2. Xuất model
export_dir = "parakeet_tdt_export"
os.makedirs(export_dir, exist_ok=True)
output_path = os.path.join(export_dir, "parakeet.onnx")

dynamic_axes = {
    "audio_signal":    {0: "audio_signal_dynamic_axes_1", 2: "audio_signal_dynamic_axes_2"},
    "length":          {0: "length_dynamic_axes_1"},
    "targets":         {0: "targets_dynamic_axes_1", 1: "targets_dynamic_axes_2"},
    "target_length":   {0: "target_length_dynamic_axes_1"},
    "encoder_outputs": {0: "encoder_outputs_dynamic_axes_1", 2: "encoder_outputs_dynamic_axes_2"},
}

model.export(
    output=output_path,
    input_example=None,
    onnx_opset_version=17,
    do_constant_folding=True,
    check_trace=False,
    verbose=True,
    dynamic_axes=dynamic_axes
)

print(f"✅ Export xong. Check file ONNX trong: {export_dir}")
