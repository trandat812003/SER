import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Config, Wav2Vec2Model

from models.wav2vec2_xlsr import SERModelWav2Vec2
from models.hubert_xlsr import SERModelHuBERT
from models.wav2vec2 import SERModel


class ONNXExporter:
    def __init__(self, model_name="wav2vec2_xlsr", model_ckpt=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Khởi tạo model
        if model_name == "wav2vec2_xlsr":
            self.model = SERModelWav2Vec2()
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        elif model_name == "hubert_xlsr":
            self.model = SERModelHuBERT()
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
        elif model_name == "wav2vec2":
            w2v_config = "/media/admin123/DataVoice/ckpt"
            checkpoint_path = "/media/admin123/DataVoice/ckpt/0-best.pth"
            w2v_config = Wav2Vec2Config.from_pretrained(w2v_config)
            w2v_model = Wav2Vec2Model(w2v_config)
            self.model = SERModel(wav2vec_model=w2v_model)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0.0,
                do_normalize=True,
                return_attention_mask=False,
            )
        else:
            raise ValueError(f"Không hỗ trợ model: {model_name}")
        
        # Load checkpoint nếu có
        if model_ckpt:
            self.model.load_state_dict(torch.load(model_ckpt, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model {model_name} đã được khởi tạo trên {self.device}")

    def create_dummy_input(self, batch_size=1, seq_length=16000):
        """Tạo dummy input cho ONNX export"""
        # Tạo dummy audio input (batch_size, seq_length)
        dummy_audio = torch.randn(batch_size, seq_length, device=self.device)
        
        return dummy_audio

    def export_to_onnx(self, output_path, batch_size=1, seq_length=16000, dynamic_axes=True):
        """Export model sang ONNX format"""
        print(f"Đang export model {self.model_name} sang ONNX...")
        
        # Tạo dummy input
        dummy_input = self.create_dummy_input(batch_size, seq_length)
        
        # Định nghĩa dynamic axes cho ONNX
        if dynamic_axes:
            dynamic_axes = {
                'input_values': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
        else:
            dynamic_axes = None
        
        # Export sang ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=['input_values'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=True
        )
        
        print(f"Model đã được export thành công: {output_path}")
        
        # Kiểm tra ONNX model
        self.verify_onnx_model(output_path, dummy_input)

    def verify_onnx_model(self, onnx_path, dummy_input):
        """Kiểm tra ONNX model"""
        try:
            import onnx
            import onnxruntime as ort
            
            # Load và kiểm tra ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model hợp lệ!")
            
            # Test inference với ONNX Runtime
            ort_session = ort.InferenceSession(onnx_path)
            
            # Chuẩn bị input cho ONNX Runtime
            ort_inputs = {
                'input_values': dummy_input.cpu().numpy()
            }
            
            # Chạy inference
            ort_outputs = ort_session.run(None, ort_inputs)
            print(f"ONNX Runtime output shape: {ort_outputs[0].shape}")
            
            # So sánh với PyTorch output
            with torch.no_grad():
                torch_output = self.model(dummy_input)
            
            torch_output_np = torch_output.cpu().numpy()
            print(f"PyTorch output shape: {torch_output_np.shape}")
            
            # Kiểm tra độ chính xác
            diff = np.abs(ort_outputs[0] - torch_output_np)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"Max difference: {max_diff:.6f}")
            print(f"Mean difference: {mean_diff:.6f}")
            
            if max_diff < 1e-5:
                print("✅ ONNX model hoạt động chính xác!")
            else:
                print("⚠️ Có sự khác biệt giữa PyTorch và ONNX output")
                
        except ImportError:
            print("⚠️ Không thể import onnx hoặc onnxruntime. Bỏ qua verification.")
        except Exception as e:
            print(f"⚠️ Lỗi khi verify ONNX model: {e}")

    def export_feature_extractor_config(self, output_dir):
        """Export feature extractor config"""
        config_path = os.path.join(output_dir, "feature_extractor_config.json")
        self.feature_extractor.save_pretrained(output_dir)
        print(f"Feature extractor config đã được lưu: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model sang ONNX")
    parser.add_argument("--model_name", type=str, default="wav2vec2_xlsr", 
                       choices=["wav2vec2_xlsr", "hubert_xlsr", "wav2vec2"],
                       help="Tên model để export")
    parser.add_argument("--model_ckpt", type=str, default=None,
                       help="Đường dẫn đến checkpoint file")
    parser.add_argument("--output_dir", type=str, default="./exported_models",
                       help="Thư mục output")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size cho dummy input")
    parser.add_argument("--seq_length", type=int, default=16000,
                       help="Sequence length cho dummy input")
    parser.add_argument("--no_dynamic_axes", action="store_true",
                       help="Không sử dụng dynamic axes")
    
    args = parser.parse_args()
    
    # Tạo thư mục output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Khởi tạo exporter
    exporter = ONNXExporter(
        model_name=args.model_name,
        model_ckpt=args.model_ckpt
    )
    
    # Export model
    output_path = os.path.join(args.output_dir, f"{args.model_name}.onnx")
    exporter.export_to_onnx(
        output_path=output_path,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        dynamic_axes=not args.no_dynamic_axes
    )
    
    # Export feature extractor config
    exporter.export_feature_extractor_config(args.output_dir)
    
    print(f"\n✅ Export hoàn thành!")
    print(f"ONNX model: {output_path}")
    print(f"Feature extractor config: {args.output_dir}/")


if __name__ == "__main__":
    main() 