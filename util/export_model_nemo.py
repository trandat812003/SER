import os
import nemo.collections.asr as nemo_asr

# 1. Load model
model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
model.eval()
model.to('cpu')  # hoặc 'cuda' nếu muốn

# 2. Tạo thư mục export
export_dir = "parakeet_tdt_export"
os.makedirs(export_dir, exist_ok=True)

# 3. Định nghĩa dynamic_axes nếu cần
dynamic_axes = {
    "input": {0: "batch_size", 1: "time_steps"},
    "output": {0: "batch_size", 1: "time_steps"}
}

# 4. Export, đặt tên file trong folder
output_path = os.path.join(export_dir, "parakeet_tdt_dynamic.onnx")

model.export(
    output=output_path,
    input_example=None,           # để NeMo tự tạo dummy input
    onnx_opset_version=13,        # khuyến nghị >=13
    do_constant_folding=True,
    check_trace=False,
    verbose=True,
    dynamic_axes=dynamic_axes
)

print(f"✅ Export completed. Check your exported files in: {export_dir}")
